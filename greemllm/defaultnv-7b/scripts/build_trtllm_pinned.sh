#!/usr/bin/env bash
set -euo pipefail

trtllm_commit=137fe35539ea182f1495f5021bfda97c729e50c3
xgrammar_commit=989222175c2a30fb7987d8bcce35bec1bf6817f2
source_dir=/home/shenhaojia/energy/.build-cache/TensorRT-LLM-src
lfs_asset=cpp/tensorrt_llm/kernels/internal_cutlass_kernels/x86_64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
lfs_asset_sha256=ee67d78be05fffe0ace420afe76513215a4a52f945df07c65b24264a3d7f4356

mapfile -t gpu_caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ')
if (( ${#gpu_caps[@]} == 0 )); then
  echo "No NVIDIA GPUs are visible; cannot select a local build architecture" >&2
  exit 1
fi
mapfile -t unique_caps < <(printf '%s\n' "${gpu_caps[@]}" | sort -u)
if (( ${#unique_caps[@]} != 1 )); then
  echo "Visible GPUs have mixed compute capabilities: ${unique_caps[*]}" >&2
  exit 1
fi
compute_cap=${unique_caps[0]}
cuda_arch=${compute_cap/./}
if [[ ! "$cuda_arch" =~ ^[0-9]+$ ]]; then
  echo "Unexpected compute capability: $compute_cap" >&2
  exit 1
fi
arch_label=sm${cuda_arch}
output_dir=/home/shenhaojia/energy/.build-cache/trtllm-v0.3.1/$arch_label
wheel_repository=docker.io/tensorrt_llm_${arch_label}
wheel_image=$wheel_repository/wheel:latest
wheel_container=greenllm-trtllm-wheel-${arch_label}

if [[ -e "$source_dir" ]]; then
  actual_commit=$(git -C "$source_dir" rev-parse HEAD)
  if [[ "$actual_commit" != "$trtllm_commit" ]]; then
    echo "$source_dir is at unexpected commit $actual_commit" >&2
    exit 1
  fi
else
  git init "$source_dir"
  git -C "$source_dir" remote add origin https://github.com/NVIDIA/TensorRT-LLM.git
  GIT_LFS_SKIP_SMUDGE=1 git -C "$source_dir" -c http.version=HTTP/1.1 \
    fetch --depth 1 origin "$trtllm_commit"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$source_dir" checkout --detach FETCH_HEAD
  git -C "$source_dir" -c http.version=HTTP/1.1 \
    submodule update --init --recursive --depth 1
fi

git -C "$source_dir" lfs pull origin \
  --include="cpp/tensorrt_llm/kernels/**" \
  --exclude="cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/**"
echo "$lfs_asset_sha256  $source_dir/$lfs_asset" | sha256sum -c

version_file="$source_dir/tensorrt_llm/version.py"
requirements_file="$source_dir/requirements.txt"
if [[ -f "$version_file.greenllm-backup" ]]; then
  mv "$version_file.greenllm-backup" "$version_file"
fi
if [[ -f "$requirements_file.greenllm-backup" ]]; then
  mv "$requirements_file.greenllm-backup" "$requirements_file"
fi
cp "$version_file" "$version_file.greenllm-backup"
cp "$requirements_file" "$requirements_file.greenllm-backup"
restore_sources() {
  if [[ -f "$version_file.greenllm-backup" ]]; then
    mv "$version_file.greenllm-backup" "$version_file"
  fi
  if [[ -f "$requirements_file.greenllm-backup" ]]; then
    mv "$requirements_file.greenllm-backup" "$requirements_file"
  fi
}
trap restore_sources EXIT
short_commit=$(git -C "$source_dir" rev-parse --short HEAD)
sed -i "s/__version__ = \"\(.*\)\"/__version__ = \"\1+dev${short_commit}\"/" "$version_file"
# xgrammar 0.1.16 was removed from PyPI. Build that same official version from
# its immutable upstream tag commit instead of substituting a newer release.
sed -i "s|^xgrammar==0.1.16$|xgrammar @ git+https://github.com/mlc-ai/xgrammar.git@${xgrammar_commit}|" \
  "$requirements_file"

# The exact NGC base image is preloaded and verified locally. Avoid forcing a
# second registry pull, since that can both fail through restricted proxies and
# silently move the mutable release tag between reproductions.
make -C "$source_dir/docker" DOCKER_BUILD_OPTS=--load \
  IMAGE_NAME="$wheel_repository" CUDA_ARCHS="${cuda_arch}-real" wheel_build

mkdir -p "$output_dir"
docker create --name "$wheel_container" "$wheel_image"
docker cp "$wheel_container":/src/tensorrt_llm/build "$output_dir/"
cp "$output_dir"/build/*.whl "$output_dir/"
docker rm "$wheel_container"
restore_sources
trap - EXIT
printf 'amd64_%s_%s\n' "$arch_label" "$trtllm_commit" > "$output_dir/commit.txt"
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader \
  > "$output_dir/build-gpus.csv"
sha256sum "$output_dir"/*.whl > "$output_dir/SHA256SUMS"
