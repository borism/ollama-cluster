# llama.cpp RPC fixes carried by this fork until upstream has them.
# See README.md in this directory.
#
# Include next to llama/compat/compat.cmake and run the command AFTER the
# compat patch command, wherever that one runs:
#
#     PATCH_COMMAND ${OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND}
#           COMMAND ${OLLAMA_LLAMA_CPP_RPC_PATCH_COMMAND}
#
# It uses the same idempotent applier, so every *.patch here is applied in
# numeric filename order and one that is already applied is skipped.

get_filename_component(_ollama_rpc_patch_applier
    "${CMAKE_CURRENT_LIST_DIR}/../../cmake/apply-git-patches.cmake"
    ABSOLUTE)

set(OLLAMA_LLAMA_CPP_RPC_PATCH_COMMAND
    ${CMAKE_COMMAND}
        -DPATCH_DIR=${CMAKE_CURRENT_LIST_DIR}
        -DPATCH_LABEL=llama/rpc-patches
        -P ${_ollama_rpc_patch_applier}
    CACHE INTERNAL "llama.cpp RPC fix patch command, run after the compat patch")
unset(_ollama_rpc_patch_applier)
