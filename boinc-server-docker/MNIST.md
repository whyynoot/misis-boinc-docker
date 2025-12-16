## MNIST demo stack

Everything below runs from `boinc-server-docker/`.

### One‑time bring‑up
```bash
# start server (Apache/MySQL) + prepared client with python/numpy
docker compose up -d

# register the MNIST app (version 1.1) inside the project
docker compose exec apache bash -lc "cd /home/boincadm/project && yes | bin/update_versions --noconfirm"

# drop the job/template files into the fresh project volume
docker cp templates/mnist_in boinc-server-docker-apache-1:/home/boincadm/project/templates/
docker cp templates/mnist_out boinc-server-docker-apache-1:/home/boincadm/project/templates/
docker cp job-sample.json boinc-server-docker-apache-1:/home/boincadm/project/download/job.json

# create a workunit that feeds the MNIST task to clients
docker compose exec apache bash -lc "cd /home/boincadm/project && \
  bin/create_work --appname mnist --wu_name mnist_clean_1 \
  --rsc_fpops_est 1e12 --rsc_fpops_bound 1e13 \
  --rsc_memory_bound 2000000000 --rsc_disk_bound 500000000 job.json"
```

The dockerised client (`client1`) auto‑attaches using the authenticator in
`docker-compose.yml` and downloads/runs the task immediately.

### What’s inside
- Custom client image `boinc/mnist-client` adds `python3`, `numpy`, `requests`.
- App files live in `apps/mnist/1.1/x86_64-pc-linux-gnu/`:
  - `mnist.py` downloads `mnist.npz`, samples the dataset, and emits digit‑wise means.
  - `mnist.sh` resolves BOINC soft links, calls the Python script, and touches `boinc_finish_called`.
- Templates: `templates/mnist_in` and `templates/mnist_out`.
- Sample job payload: `job-sample.json` (seed + sample_count).

### Inspecting results
Returned metrics land under `upload/*/mnist_clean_1_*`. Example:
`docker compose exec apache cat /home/boincadm/project/upload/299/mnist_clean_1_0_*`

### Notes
- Account authenticator baked into the compose file: `89df5afee44433452a278a4309a1ccb6`.
  Regenerate with a temporary client if you want a new one, then update `BOINC_CMD_LINE_OPTIONS`.
- If you `docker compose down -v`, rerun the commands in the bring‑up section
  to repopulate templates/app files and submit a fresh workunit.
