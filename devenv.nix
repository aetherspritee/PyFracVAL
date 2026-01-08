{
  pkgs,
  config,
  ...
}:

{
  env.GREET = "devenv";

  packages = with pkgs; [
    git
    beads
  ];

  # shell = lib.mkForce pkgs.fish;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  enterShell = ''
    hello
    git --version
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
    uv run pyfracval --version
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/

  languages.python = {
    enable = true;
    # version = "3.12";

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

    libraries = [ pkgs.zlib ];
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

    bd-sync = {
      enable = true;
      description = "beads sync";
      package = pkgs.beads;
      pass_filenames = false;
      entry = "bd sync --flush-only";
    };

    bd-pre-push = {
      enable = true;
      description = "beads pre-push check";
      package = pkgs.beads;
      stages = [ "pre-push" ];
      pass_filenames = false;
      entry = ''
        bash -c '
        bd sync --flush-only
        if [ -n "$(git status --porcelain .beads/*.jsonl)" ]; then
            echo "Error: .beads JSONL files have uncommitted changes. Please commit them before pushing."
            exit 1
        fi
        '
      '';
    };

    bd-post-merge = {
      enable = true;
      description = "beads import";
      package = pkgs.beads;
      stages = [ "post-merge" ];
      pass_filenames = false;
      entry = "bd import -i .beads/issues.jsonl";
    };

    bd-post-checkout = {
      enable = true;
      description = "beads import on checkout";
      package = pkgs.beads;
      stages = [ "post-checkout" ];
      pass_filenames = false;
      entry = ''
        bash -c '
        # Args: $1 old HEAD, $2 new HEAD, $3 1=branch checkout, 0=file checkout
        if [ "$3" != "1" ]; then
            exit 0
        fi

        # During rebase, git checks out commits; must not modify worktree.
        if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then
            exit 0
        fi

        if [ ! -d .beads ]; then
            exit 0
        fi

        if ! ls .beads/*.jsonl >/dev/null 2>&1; then
            exit 0
        fi

        if ! output=$(bd sync --import-only --no-git-history 2>&1); then
            echo "Warning: Failed to sync bd changes after checkout" >&2
            echo "$output" >&2
            echo "" >&2
            echo "Run bd doctor --fix to diagnose and repair" >&2
        fi

        bd doctor --check-health 2>/dev/null || true
        '
      '';
    };
  };
}
