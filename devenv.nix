{
  lib,
  pkgs,
  config,
  ...
}:

{
  env = {
    GREET = "devenv";
    UV_PYTHON = toString config.languages.python.package.interpreter;
    LD_LIBRARY_PATH = lib.makeLibraryPath config.languages.python.libraries;

    VTK_USE_X = "OFF";
    VTK_DEFAULT_OPENGL_WINDOW = "vtkEGLRenderWindow";
    # VTK_DEFAULT_OPENGL_WINDOW = "vtkOSOpenGLRenderWindow";

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

    libraries = with pkgs; [
      zlib
      stdenv.cc.cc.lib
      libGL
    ];
  };

  git-hooks.hooks = {
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
