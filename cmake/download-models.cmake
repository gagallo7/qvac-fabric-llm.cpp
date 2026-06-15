get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

# Split the "ALGO=value" hash spec so we can verify the download ourselves.
string(REPLACE "=" ";" hash_parts "${HASH}")
list(GET hash_parts 0 hash_algo)
list(GET hash_parts 1 hash_value)

# Reuse an already-downloaded, valid copy.
if(EXISTS "${DEST}")
    file(${hash_algo} "${DEST}" have_hash)
    if(have_hash STREQUAL hash_value)
        return()
    endif()
endif()

# Source URL defaults to the ggml-org HF repo but can be overridden, e.g. to a
# mirror/proxy, or to httpbin when testing the retry path.
if(NOT DEFINED MODEL_URL)
    set(MODEL_URL "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true")
endif()
message(STATUS "Downloading ${NAME} from ${MODEL_URL}...")

# Delegate the transfer (and its retries/backoff) to curl or wget instead of
# hand-rolling a retry loop around file(DOWNLOAD).
find_program(CURL NAMES curl)
find_program(WGET NAMES wget)
if(CURL)
    execute_process(
        COMMAND "${CURL}" --location --fail --retry 5 --retry-all-errors
                --connect-timeout 30 --output "${DEST}" "${MODEL_URL}"
        RESULT_VARIABLE rc)
elseif(WGET)
    execute_process(
        COMMAND "${WGET}" --tries=5 --waitretry=2 --timeout=30
                -O "${DEST}" "${MODEL_URL}"
        RESULT_VARIABLE rc)
else()
    # No retry-capable downloader on this runner; fall back to a single
    # cmake-native attempt (the original upstream behaviour).
    message(STATUS "curl/wget not found; falling back to file(DOWNLOAD)")
    file(DOWNLOAD "${MODEL_URL}" "${DEST}" TLS_VERIFY ON STATUS status)
    list(GET status 0 rc)
endif()

if(NOT rc EQUAL 0)
    file(REMOVE "${DEST}")
    message(FATAL_ERROR "Failed to download ${NAME} (downloader exit ${rc})")
endif()

file(${hash_algo} "${DEST}" have_hash)
if(NOT have_hash STREQUAL hash_value)
    file(REMOVE "${DEST}")
    message(FATAL_ERROR "Hash mismatch for ${NAME}: expected ${hash_value}, got ${have_hash}")
endif()
