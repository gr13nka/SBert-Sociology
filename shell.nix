with import <nixpkgs> { config.allowUnfree = true; };
with python310Packages;
stdenv.mkDerivation {
  name = "aicdw-cuda-shell";
  buildInputs = [
    stdenv.cc.cc.lib
    pkgs.python310Full
    python310Packages.pip
    python310Packages.virtualenv
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ]}
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib/:$LD_LIBRARY_PATH

    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install -r requirements.txt
    export PATH=$PWD/venv/bin:$PATH
    export PYTHONPATH=venv/lib/python3.10/site-packages/:$PYTHONPATH
  '';

  postShellHook = ''
    ln -sf PYTHONPATH/* venv/lib/python3.10/site-packages
  '';
}
