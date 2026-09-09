"""Resolve an explicit NCCL runtime from a library-search path list."""

import os


def configure_nccl_library(environment, raw_directories):
    directories = [os.path.abspath(item) for item in
                   raw_directories.split(os.pathsep) if item]
    if not directories:
        raise ValueError("--nccl-lib-dir did not contain a directory")
    current = [item for item in environment.get("LD_LIBRARY_PATH", "").split(
        os.pathsep) if item]
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(directories + current)
    candidates = [os.path.join(directory, "libnccl.so.2")
                  for directory in directories]
    library = next((candidate for candidate in candidates
                    if os.path.isfile(candidate)), None)
    if library is None:
        raise FileNotFoundError(
            "libnccl.so.2 was not found in --nccl-lib-dir entries: " +
            ", ".join(directories))
    environment["WEFTMOE_NCCL_LIBRARY"] = library
    return library
