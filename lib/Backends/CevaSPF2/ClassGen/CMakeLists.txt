set(VERIF_FILENAME CevaSPF2SpecificNodesVerification.h)
configure_file(${VERIF_FILENAME}
               ${GLOW_BINARY_DIR}/glow/${VERIF_FILENAME} COPYONLY)

set(VERIF_FILENAME CevaSPF2SpecificInstrsVerification.h)
configure_file(${VERIF_FILENAME}
               ${GLOW_BINARY_DIR}/glow/${VERIF_FILENAME} COPYONLY)
