## CIFAR-100 demo stack

Everything below runs from `boinc-server-docker/`.

### One-time bring-up
```bash
# start server (Apache/MySQL) + clients with PyTorch/torchvision
docker compose up -d

# register the CIFAR-100 app (version 1.0) inside the project
docker compose exec apache bash -lc "cd /home/boincadm/project && yes | bin/update_versions --appname cifar100 --noconfirm"

# drop the job/template files into the fresh project volume
docker cp templates/cifar100_in boinc-server-docker-apache-1:/home/boincadm/project/templates/
docker cp templates/cifar100_out boinc-server-docker-apache-1:/home/boincadm/project/templates/
docker cp job-sample-cifar100.json boinc-server-docker-apache-1:/home/boincadm/project/download/job.json

# create a workunit that feeds the CIFAR-100 task to clients
docker compose exec apache bash -lc "cd /home/boincadm/project && \
  bin/create_work --appname cifar100 --wu_name cifar100_clean_1 \
  --wu_template templates/cifar100_in --result_template templates/cifar100_out \
  --rsc_fpops_est 2e12 --rsc_fpops_bound 5e13 \
  --rsc_memory_bound 4000000000 --rsc_disk_bound 2000000000 job.json"
```

Clients (`client1`..`client10`) auto-attach using the authenticator in
`docker-compose.yml` and will download/run the task as soon as it is available.

### What's inside
- Custom client image `boinc/mnist-client` now adds `python3`, `numpy`, `requests`, `torch`, `torchvision`.
- App files live in `apps/cifar100/1.0/x86_64-pc-linux-gnu/`:
  - `cifar100.py` downloads CIFAR-100, trains ConvNeXt-Tiny, validates on the test set, and emits metrics.
  - `cifar100.sh` resolves BOINC soft links, calls the Python script, and touches `boinc_finish_called`.
- Templates: `templates/cifar100_in` and `templates/cifar100_out`.
- Sample job payload: `job-sample-cifar100.json`.

### Inspecting results
Returned metrics land under `upload/*/cifar100_clean_1_*`. Example:
`docker compose exec apache cat /home/boincadm/project/upload/299/cifar100_clean_1_0_*`

### Notes
- If you change the app code, rerun `bin/update_versions --appname cifar100` and reset clients:
  `docker compose exec client1 boinccmd --project http://host.docker.internal:8082/boincserver reset`
- If you `docker compose down -v`, rerun the commands in the bring-up section
  to repopulate templates/app files and submit a fresh workunit.
