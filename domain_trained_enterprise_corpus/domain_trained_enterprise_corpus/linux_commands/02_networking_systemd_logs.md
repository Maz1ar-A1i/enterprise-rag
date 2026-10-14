# Linux Commands - 02 Networking Systemd Logs

> Corpus role: enterprise infrastructure knowledge for the Domain-Trained Enterprise RAG capstone. Content is intentionally modular so it can be chunked into independent retrieval units.

## Topic scope

Topic: **linux_commands**

Each section below is an atomic knowledge unit. Preserve the heading and topic metadata during ingestion so retrieval results retain their semantic context.

## LINUX_COMMANDS-001 | IP and route inspection

`ip addr` shows interfaces and addresses; `ip route` shows routing information. When a service cannot reach a dependency, check DNS, route, interface state and listening ports before assuming the application is broken.

## LINUX_COMMANDS-002 | Listening sockets

`ss -lntup` is a common way to inspect listening TCP/UDP sockets and associated processes where permitted. Confirm the address and port the process binds to. A process listening on `127.0.0.1` is not reachable through the host's external interface.

## LINUX_COMMANDS-003 | DNS troubleshooting

`getent hosts`, `resolvectl` where available, and tools such as `dig` or `nslookup` can help diagnose name resolution. DNS problems often appear as application connection timeouts, so distinguish name-resolution errors from TCP connection errors.

## LINUX_COMMANDS-004 | curl as an HTTP diagnostic

`curl -v` can show DNS resolution, TCP/TLS connection steps, request headers and response headers. It is useful for testing NGINX, FastAPI health endpoints and internal Services. Avoid sending credentials or tokens in command history on shared systems.

## LINUX_COMMANDS-005 | systemd services

`systemctl status`, `systemctl start`, `systemctl stop`, `systemctl restart` and `systemctl enable` manage systemd services. A service unit describes how a process is launched and supervised. After changing a unit file, reload the systemd manager configuration as appropriate before restarting the service.

## LINUX_COMMANDS-006 | journalctl

`journalctl` reads systemd journal entries. Use time, unit and priority filters to narrow the investigation. Logs are evidence of what the service reported, not proof that the reported root cause is correct; correlate with metrics and dependency logs.

## LINUX_COMMANDS-007 | CPU and memory pressure

Use `uptime`, `top`, `vmstat` and `free` to inspect system pressure. High load average can reflect CPU saturation, blocked I/O or runnable tasks, so it should not be interpreted as CPU percentage by itself.

## LINUX_COMMANDS-008 | Disk pressure

Use `df -h` to check filesystem capacity and `du -xh` to locate large directories. Also consider inode exhaustion. Containers, logs and model files can consume disk even when the application database itself appears healthy.

## LINUX_COMMANDS-009 | Log rotation

Unbounded logs can fill disks. Configure a retention policy and rotation mechanism appropriate to the operating system/container environment. Centralized logging is preferable for multi-node systems, but the local node must still prevent runaway storage use.

## LINUX_COMMANDS-010 | Resource limits and isolation

Linux namespaces and control groups underpin important forms of process isolation and resource control used by container platforms. Understanding these primitives helps explain why CPU, memory, filesystem and network behavior in containers differs from a simple process on the host.

## Source basis

The corpus is intentionally grounded in official project documentation. Use the URLs below for version-specific verification rather than copying volatile version numbers into model-training text.

- https://www.gnu.org/software/coreutils/manual/
- https://www.gnu.org/software/coreutils/manual/html_node/Mode-Structure.html
- https://kernel.org/doc/html/latest/

## Versioning note

This document should be treated as a versioned corpus artifact. Record the source snapshot date and any later updates in the dataset manifest. Version-sensitive benchmark questions should state the relevant software version explicitly.
