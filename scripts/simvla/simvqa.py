"""Compatibility entry point; the CPU converter is distributed in the simvla wheel."""
from simvla.vqa import *  # noqa: F401,F403
from simvla.vqa import main

if __name__ == "__main__":
    main()
