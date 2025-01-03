# Sources:
# https://pyproject-nix.github.io/pyproject.nix/use-cases/pyproject.html
# https://pyproject-nix.github.io/uv2nix/usage/hello-world.html

{
  description = "PyFracVAL flake devenv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
    };
  };

  outputs =
    inputs@{
      flake-parts,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];
      perSystem =
        {
          self',
          pkgs,
          lib,
          ...
        }:
        let
          project = pyproject-nix.lib.project.loadPyproject { projectRoot = ./.; };
          workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

          overlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };

          pyprojectOverrides = final: prev: {
            numba = prev.numba.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.tbb_2021_11 ];
            });
            taichi = prev.taichi.overrideAttrs (old: {
              # buildInputs =
              #   (old.buildInputs or [ ])
              #   ++ (with pkgs; [
              #     libGL
              #     vulkan-loader
              #   ]);
              propagatedBuildInputs =
                (old.propagatedBuildInputs or [ ])
                ++ (with pkgs; [
                  libGL
                  vulkan-loader
                ]);
              pythonImportsCheck = "OpenGL";
            });
          };

          python = pkgs.python3;

          pythonSet =
            (pkgs.callPackage pyproject-nix.build.packages {
              inherit python;
            }).overrideScope
              (
                lib.composeManyExtensions [
                  pyproject-build-systems.overlays.default
                  overlay
                  pyprojectOverrides
                ]
              );

        in
        {
          packages = {
            default = self'.packages.pyfracval;
            pyfracval = python.pkgs.buildPythonPackage (
              (project.renderers.buildPythonPackage { inherit python; }) // { env.CUSTOM_ENVVAR = "hello"; }
            );
            pyfracval-env = pythonSet.mkVirtualEnv "pyfracval-env" workspace.deps.default;
          };

          devShells = {
            default =
              let
                arg = project.renderers.withPackages {
                  inherit python;
                  extras = [
                    # "test"
                  ];
                };

                pythonEnv = python.withPackages arg;
              in
              pkgs.mkShell {
                packages = [
                  pythonEnv
                  self'.packages.pyfracval
                ];
              };

            impure = pkgs.mkShell {
              packages = [
                python
                pkgs.uv
              ];
              shellHook = ''
                unset PYTHONPATH
              '';
            };

            uv2nix =
              let
                editableOverlay = workspace.mkEditablePyprojectOverlay {
                  root = "$REPO_ROOT";
                };

                editablePythonSet = pythonSet.overrideScope editableOverlay;

                virtualenv = editablePythonSet.mkVirtualEnv "pyfracval-dev-env" (
                  workspace.deps.default
                  // {
                    pyfracval = [
                      # "explore"
                      # "test"
                    ];
                  }
                );

              in
              pkgs.mkShell {
                packages = [
                  virtualenv
                  pkgs.uv
                ];
                shellHook = ''
                  unset PYTHONPATH
                  export REPO_ROOT=$(git rev-parse --show-toplevel)
                  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/opengl-driver/lib:${
                    lib.makeLibraryPath (
                      with pkgs;
                      [
                        libGL
                        vulkan-loader
                        xorg.libX11
                      ]
                    )
                  }
                '';
              };
          };
        };
    };
}
