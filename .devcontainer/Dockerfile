ARG BASE=mcr.microsoft.com/devcontainers/base:ubuntu-20.04

ARG LLVM_VERSION=15.0.7
ARG HETEROCL_VERSION=69181b080c8cf7d9fab0a6b2ae3511923274b32e
ARG HCL_DIALECT_VERSION=f2f5e43a93173f9a86671c8f5004d5fb21cb329b

FROM quay.io/pypa/manylinux_2_28_x86_64 AS build-mlir

## LLVM 15 ##

ARG PYTHON_DIR=/opt/python/cp38-cp38
ARG LLVM_DIR=/opt/llvm-project

ARG PYTHON=$PYTHON_DIR/bin/python3

# Install build dependencies
RUN yum install -y \
    jq

# Get latest LLVM 15 release
# Get latest LLVM 15 release
RUN mkdir /tmp/llvm-project && \
    curl -s "https://api.github.com/repos/llvm/llvm-project/releases?per_page=100" | \
        jq -r 'sort_by(.published_at) | map(select(.tag_name | startswith("llvmorg-15"))) | last' > /tmp/llvm-project/release.json && \
    jq -r '.tarball_url' /tmp/llvm-project/release.json > /tmp/llvm-project/url && \
    jq -r '.tag_name' /tmp/llvm-project/release.json > /tmp/llvm-project/tag && \
    mkdir $LLVM_DIR && \
    curl -fLs "$(cat /tmp/llvm-project/url)" | tar -xz --strip=1 -C $LLVM_DIR

WORKDIR $LLVM_DIR/build
RUN env PATH="$PYTHON_DIR/bin:$PATH" \
        pip install -r ../mlir/python/requirements.txt && \
    cmake ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=$PYTHON_DIR \
        -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLVM_BUILD_BENCHMARKS=OFF \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_BUILD_RUNTIMES=OFF \
        -DLLVM_BUILD_TESTS=OFF \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_RUNTIMES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_TOOLS=ON \
        -DLLVM_INCLUDE_UTILS=ON \
        -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_TARGETS_TO_BUILD=Native \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE="$PYTHON" \
        && \
    make -j`nproc` && \
    mkdir ../out && \
    cp -r bin include lib tools/mlir/python_packages ../out

WORKDIR /tmp/wheel-factory/mlirlib
RUN mv $LLVM_DIR/out mlirlib && \
    touch mlirlib/__init__.py && \
    cat | sed "s/DEADBEEF/$(grep -o '[0-9\.]\+' /tmp/llvm-project/tag)/" <<EOF > pyproject.toml
[project]
name = "mlirlib"
version = "DEADBEEF"
requires-python = ">= 3.8"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
ignore-vcs = true

[tool.hatch.build.targets.wheel]
pure_python = false
packages = ["mlirlib"]
EOF

RUN $PYTHON -m build --wheel -o /opt/wheelhouse

#&& \
#    auditwheel repair -w /opt/wheelhouse dist/*.whl

FROM $BASE AS build

ARG LLVM_VERSION
ARG HETEROCL_VERSION
ARG HCL_DIALECT_VERSION
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        apt-transport-https \
        ca-certificates \
        g++ \
        gcc \
        git \
        gnupg \
        make \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-virtualenv \
        software-properties-common \
        wget \
        xz-utils
        
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update -y && \
    apt-get install -y cmake

RUN pip install -U \
        pip \
        build \
        patchelf \
        auditwheel

# Download LLVM
ARG LLVM_DIR=/opt/llvm-project
ADD "https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/llvm-project-$LLVM_VERSION.src.tar.xz" /tmp/
RUN cd /tmp && \
    tar -xf llvm-project-$LLVM_VERSION.src.tar.xz && \
    mv llvm-project-$LLVM_VERSION.src $LLVM_DIR && \
    rm llvm-project-$LLVM_VERSION.src.tar.xz

# Build LLVM & MLIR
WORKDIR $LLVM_DIR/build
RUN pip install -r ../mlir/python/requirements.txt
RUN cmake ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD=Native \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INSTALL_UTILS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        && \
    make -j`nproc`

# Download HeteroCL
ARG HETEROCL_DIR=/opt/heterocl
ARG HCL_DIALECT_DIR=$HETEROCL_DIR/hcl-dialect
RUN git clone https://github.com/cornell-zhang/heterocl.git $HETEROCL_DIR && \
    cd $HETEROCL_DIR && \
    git checkout $HETEROCL_VERSION
RUN git clone https://github.com/cornell-zhang/hcl-dialect.git $HCL_DIALECT_DIR && \
    cd $HCL_DIALECT_DIR && \
    git checkout $HCL_DIALECT_VERSION

# Build hcl-dialect
WORKDIR $HCL_DIALECT_DIR/build
RUN cmake .. \
        -DMLIR_DIR=$LLVM_DIR/build/lib/cmake/mlir \
        -DLLVM_EXTERNAL_LIT=$LLVM_DIR/build/bin/llvm-lit \
        -DPYTHON_BINDING=ON \
        -DOPENSCOP=OFF \
        && \
    make -j`nproc`

# Package hcl-dialect
WORKDIR $HCL_DIALECT_DIR/build/tools/hcl/python_packages/hcl_core
RUN rm setup.py && \
    cat <<EOF > pyproject.toml
[project]
name = "hcl_mlir"
requires-python = ">= 3.6"
version = "0.1"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
ignore-vcs = true

[tool.hatch.build.targets.wheel]
pure_python = false
packages = ["hcl_mlir"]
EOF
RUN python3 -m build --wheel -o /opt/wheelhouse

# Package HeteroCL
WORKDIR $HETEROCL_DIR
RUN rm setup.py && \
    cat <<EOF > pyproject.toml
[project]
name = "heterocl"
requires-python = ">= 3.6"
version = "0.5"
dynamic = ["dependencies"]

[build-system]
requires = ["hatchling", "hatch-requirements-txt"]
build-backend = "hatchling.build"

[tool.hatch.build]
ignore-vcs = true

[tool.hatch.build.targets.wheel]
pure_python = false
packages = ["heterocl"]

[tool.hatch.metadata.hooks.requirements_txt]
files = ["requirements.txt"]
EOF
RUN python3 -m build --wheel -o /opt/wheelhouse

FROM $BASE

ARG DEBIAN_FRONTEND=noninteractive

COPY --from=build /opt/wheelhouse /opt/wheelhouse/
COPY --from=build-mlir /opt/wheelhouse /opt/wheelhouse/

RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip install -U pip attrs && \
    pip install /opt/wheelhouse/*.whl && \
    rm -r /var/lib/apt/lists/* /opt/wheelhouse

RUN pip install ipywidgets plotly matplotlib numpy

RUN MLIR_PATH=$(python3 -c "import mlirlib; print(*mlirlib.__path__)") && \
    cat <<EOF >> /home/vscode/.profile
        
if [ -d "$MLIR_PATH/bin" ] ; then
    PATH="$MLIR_PATH/bin:\$PATH"
fi
EOF