{
  lib,
  pkgs,
  config,
  ...
}:
let
  cuda = lib.getDev (
    pkgs.symlinkJoin {
      name = "cudatoolkit";
      paths = with pkgs.cudaPackages_12; [
        cudatoolkit
        "${cuda_nvcc}/nvvm"
        (lib.getStatic cuda_cudart)
      ];
      postBuild = ''
        ln -s $out/lib $out/lib64
      '';
    }
  );
in
{
  env = {
    GREET = "devenv";
    UV_PYTHON = toString config.languages.python.package.interpreter;
    # "/run/opengl-driver/lib" (the NVIDIA userspace driver, incl. libcuda.so)
    # is a raw directory, not a package derivation - append it directly
    # rather than through makeLibraryPath (which appends "/lib" to each
    # entry and would otherwise mangle a full file path into a bogus one).
    LD_LIBRARY_PATH = "${lib.makeLibraryPath config.languages.python.libraries}:/run/opengl-driver/lib";

    VTK_USE_X = "OFF";
    VTK_DEFAULT_OPENGL_WINDOW = "vtkEGLRenderWindow";
    # VTK_DEFAULT_OPENGL_WINDOW = "vtkOSOpenGLRenderWindow";

    # GPU (CUDA) support for numba's CUDA target and JAX's cuda12 plugin.
    # Mirrors ../YASF-new/devenv.nix's setup.
    CUDA_HOME = cuda;
    NUMBA_CUDA_DRIVER = "/run/opengl-driver/lib/libcuda.so";
    NUMBA_DISABLE_INTEL_SVML = true;

    # Persistent JIT caches - both frameworks recompile kernels from scratch
    # on every process start otherwise, which dominates short benchmark runs.
    NUMBA_CACHE_DIR = "${config.devenv.state}/numba_cache";
    JAX_COMPILATION_CACHE_DIR = "${config.devenv.state}/jax_cache";
  };

  packages = with pkgs; [
    git
  ];

  enterShell = ''
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
  '';

  # https://devenv.sh/tasks/
  tasks = {
    "docs:build".exec = "devenv shell -- uv run sphinx-build -b html docs/source docs/build/html";
    "docs:strict".exec =
      "devenv shell -- uv run sphinx-build -b html -W -n --keep-going docs/source docs/build/html";
    "docs:linkcheck".exec =
      "devenv shell -- uv run sphinx-build -b linkcheck docs/source docs/build/linkcheck";
    "docs:check" = {
      exec = "bash scripts/check-docs.sh ci";
      after = [ "docs:build" ];
    };
  };

  languages.python = {
    enable = true;

    uv = {
      enable = true;
      sync = {
        enable = true;
        groups = [
          "test"
          "docs"
        ];
      };
    };

    libraries = [
      cuda
    ]
    ++ (with pkgs; [
      zlib
      stdenv.cc.cc.lib
      libGL
    ]);
  };

  git-hooks.hooks = {
    sync-docs-requirements = {
      enable = true;
      description = "Regenerate docs/requirements.txt from the docs dependency group - CI's sphinx-notes/pages action installs from this file, not pyproject.toml, so it must stay in sync manually otherwise.";
      package = pkgs.uv;
      entry = "uv export --only-group docs --no-hashes -o docs/requirements.txt";
      pass_filenames = false;
      files = "^(pyproject\\.toml|uv\\.lock)$";
    };

    isort = {
      enable = true;
      settings.profile = "black";
    };

    ruff-format = {
      enable = true;
      description = "Ruff formatter";
      package = config.git-hooks.tools.ruff;
      entry = "ruff format";
      types = [ "python" ];
      args = [ "--check" ];
      after = [ "isort" ];
    };

    # ruff-check = {
    #   enable = true;
    #   description = "Ruff linter";
    #   package = config.git-hooks.tools.ruff;
    #   entry = "ruff check";
    #   types = [ "python" ];
    #   # args = [ "--fix" ];
    #   after = [ "ruff-format" ];
    # };

    # ty = {
    #   enable = true;
    #   description = "ty type check";
    #   package = pkgs.ty;
    #   entry = "ty check .";
    #   pass_filenames = false;
    #   types = [ "python" ];
    #   after = [ "ruff-check" ];
    #   require_serial = true;
    # };
  };
}
