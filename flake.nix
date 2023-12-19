{
  description = "flake template";

  inputs = {
    flake-lock.url = "github:wrvsrx/flake-lock";
    nixpkgs.follows = "flake-lock/nixpkgs";
    flake-parts.follows = "flake-lock/flake-parts";
    nur-wrvsrx = {
      url = "github:wrvsrx/nur-packages";
      inputs.flake-lock.follows = "flake-lock";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-parts.follows = "flake-parts";
    };
  };

  outputs = inputs': inputs'.flake-parts.lib.mkFlake { inputs = inputs'; } ({ inputs, ... }: {
    systems = [ "x86_64-linux" ];
    perSystem = { pkgs, system, ... }: {
      _module.args.pkgs = import inputs.nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [
          inputs.nur-wrvsrx.overlays.default
        ];
      };
      devShells.default = pkgs.callPackage ./shell.nix { };
      formatter = pkgs.nixpkgs-fmt;
    };
  });
}
