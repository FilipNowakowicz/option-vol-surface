{
  description = "Options pricer, implied vol, vol surface, Greek attribution";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python3.withPackages (ps: with ps; [
            numpy
            pandas
            scipy
            matplotlib
            yfinance
            pyarrow
            pytest
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.ruff
              pkgs.basedpyright
            ];
          };
        }
      );
    };
}
