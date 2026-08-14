if(NOT DEFINED SOURCE_DIR OR NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "SOURCE_DIR and PATCH_FILE are required")
endif()

execute_process(
  COMMAND git apply --check "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE patch_can_apply
  OUTPUT_QUIET
  ERROR_QUIET
)

if(patch_can_apply EQUAL 0)
  execute_process(
    COMMAND git apply --whitespace=nowarn "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE patch_result
  )
  if(NOT patch_result EQUAL 0)
    message(FATAL_ERROR "Failed to apply ${PATCH_FILE}")
  endif()
else()
  execute_process(
    COMMAND git apply --reverse --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE patch_already_applied
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT patch_already_applied EQUAL 0)
    message(FATAL_ERROR
      "${PATCH_FILE} neither applies cleanly nor appears to be applied")
  endif()
  message(STATUS "Patch already applied: ${PATCH_FILE}")
endif()
