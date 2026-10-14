# Linux Commands - 01 Files Permissions Processes

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **linux_commands**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## LINUX_COMMANDS-001 | Filesystem navigation

Core commands such as `pwd`, `ls`, `cd`, `find`, `stat`, `du` and `df` provide the basic vocabulary for investigating Linux systems. `df` describes filesystem capacity; `du` describes space consumed by files/directories. A full filesystem and a large directory are related but not identical diagnoses.

## LINUX_COMMANDS-002 | File permissions

Linux permissions distinguish read, write and execute for owner, group and other users. For regular files, execute permits running the file as a program; for directories, read permits listing entries, write permits creating/removing entries, and execute permits accessing entries. Permissions should be interpreted in the context of ownership and the directory path.

## LINUX_COMMANDS-003 | chmod and symbolic permissions

`chmod` changes permission bits. Symbolic forms such as `chmod u+x script.sh` express the intended change clearly. Numeric forms such as `chmod 640 file` set owner/group/other permission bits directly. Avoid `chmod -R 777` as a generic fix because it removes meaningful access control.

## LINUX_COMMANDS-004 | Ownership and groups

`chown` changes file ownership and `chgrp` changes group ownership. Group-based permissions are useful for shared application directories. Service users should receive access through explicit group membership or ACLs rather than broad world-write permissions.

## LINUX_COMMANDS-005 | Processes and signals

`ps`, `top`, `pgrep` and `pstree` help inspect processes. `kill` sends signals; `SIGTERM` requests graceful shutdown, while `SIGKILL` forces termination when a process cannot or will not exit. Prefer graceful signals for services so they can close files and connections.

## LINUX_COMMANDS-006 | Environment and PATH

`env`, `printenv`, `export` and `echo $PATH` help inspect environment variables. Environment configuration can change how software resolves commands and libraries. For reproducibility, record relevant environment variables but do not leak secrets into logs or experiment metadata.

## LINUX_COMMANDS-007 | Redirection and pipelines

Shell pipelines connect the standard output of one command to the standard input of another. `>`, `>>`, `<` and `2>` control redirection. A pipeline can hide errors when only the last command's status is checked; scripts should use appropriate shell error-handling options and explicit checks.

## LINUX_COMMANDS-008 | grep, sed and awk

`grep` searches text; `sed` performs stream editing; `awk` is useful for field-oriented processing. These tools are valuable for logs and configuration inspection, but avoid complex one-liners in production runbooks when a clearer script is safer.

## LINUX_COMMANDS-009 | archives and compression

`tar` packages files and can combine with compression tools such as gzip or zstd depending on the environment. Preserve file ownership and permissions when creating backups that are intended for system restoration, and validate the archive before trusting it.

## LINUX_COMMANDS-010 | system information

`uname`, `lscpu`, `free`, `lsblk`, `ip`, `ss` and related commands expose CPU, memory, storage and network state. Performance experiments should capture hardware/runtime information so model and retrieval latency can be interpreted correctly.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://www.gnu.org/software/coreutils/manual/
- https://www.gnu.org/software/coreutils/manual/html_node/Mode-Structure.html
- https://kernel.org/doc/html/latest/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
