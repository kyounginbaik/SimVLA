# Source from a launcher. Sets SIMVLA_KIT_ROOT_ARGS for --kit_args (and SIMVLA_KIT_ROOT).
#
# Omniverse Kit otherwise runs "portable" with its root inside the simulator venv, so every
# job on every node shares one DerivedDataCache. On Lustre, two jobs starting together on
# different nodes have hung in sync I/O during isaacsim.core.simulation_manager startup.
# Default: one root per user per node on local disk (same-node jobs share it via Kit's
# per-instance locks; the first job on a node recompiles shaders, a few minutes).
# SIMVLA_KIT_PORTABLE_ROOT=/abs/path chooses another root; =none keeps Kit's default.
SIMVLA_KIT_ROOT_ARGS=""
SIMVLA_KIT_ROOT=""
if [[ ${SIMVLA_KIT_PORTABLE_ROOT:-} != none ]]; then
  SIMVLA_KIT_ROOT=${SIMVLA_KIT_PORTABLE_ROOT:-${TMPDIR:-/tmp}/simvla-kit-${USER:-$(id -un)}}
  if [[ $SIMVLA_KIT_ROOT != /* || $SIMVLA_KIT_ROOT =~ [[:space:]] ]]; then
    echo "SIMVLA_KIT_PORTABLE_ROOT must be an absolute path without whitespace, or 'none'" >&2
    exit 2
  fi
  if [[ ${SIMVLA_DRY_RUN:-0} != 1 ]]; then
    mkdir -p "$SIMVLA_KIT_ROOT"
  fi
  SIMVLA_KIT_ROOT_ARGS=" --portable-root $SIMVLA_KIT_ROOT"
fi
