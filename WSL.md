# Running vLLM on Windows (with WSL)

vLLM, the server that runs an open model on our own GPU, works only on Linux. On Windows, you run it inside **WSL** (Windows Subsystem for Linux): a small Ubuntu inside Windows. It doesn't touch your files, or another Ubuntu you may have installed alongside Windows.

The MathEasy page itself still runs on Windows as usual. Only the vLLM server runs in Ubuntu, and Windows reaches it at `localhost:8000`. To find out why we added vLLM and how it compares with Gemini, see [docs/vllm.md](docs/vllm.md).

**What you need:**
- an NVIDIA GPU, with the normal Windows NVIDIA driver installed (WSL uses it);
- about **12 GB free on C:**: about 9 GB for vLLM and 2 GB for the model;
- 30-60 minutes the first time, mostly downloads.

---

## 1. Install WSL and Ubuntu (once)

Open **Command Prompt as administrator**: click Start, type `cmd`, right-click it and choose **Run as administrator**. Then run:

```
wsl --install --web-download
```

**Restart the PC** when it finishes.

If you get **"Forbidden (403)"**, run these instead:
```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
Restart the PC, open an admin Command Prompt again, and run:
```
wsl --update --web-download
wsl --install -d Ubuntu --web-download
```

The `--web-download` part matters: the plain command downloads from the Microsoft Store, which can be blocked.

When the **Ubuntu** window opens, it asks for a username and password; choose anything simple. The password stays invisible while you type.

## 2. Check that Ubuntu can see the GPU

In the Ubuntu window, run:
```
cd ~
nvidia-smi
```
You should see a table with your GPU's name, for example "RTX 4050". If you don't, update your Windows NVIDIA driver.

## 3. Install vLLM (once)

In the Ubuntu window, run these one at a time:
```
sudo apt update && sudo apt install -y python3-pip python3-venv build-essential
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
mkdir -p ~/tmp
TMPDIR=~/tmp pip install --no-cache-dir --timeout 30 --retries 20 --resume-retries 10 vllm
rm -rf ~/tmp
```

**Why the extra options:**
- `build-essential` installs the C compiler that vLLM needs when it starts.
- `TMPDIR=~/tmp` is needed because Ubuntu's own temporary folder lives in memory and is too small for vLLM's big files.
- `--no-cache-dir` stops pip from keeping a second copy of every file.
- `--resume-retries` lets a stalled download pick up where it stopped.

## 4. Start the vLLM server

From **Windows PowerShell**, one command:
```
wsl -d Ubuntu -- bash -lc "source ~/vllm-env/bin/activate && VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.6 --generation-config vllm"
```

The first time, it downloads the model (about 2 GB). After that, it takes 1-2 minutes to start. It's ready when it prints **`Application startup complete`**. Leave this window open: closing it stops the server.

**About the options:**
- **Qwen2.5-3B-Instruct-AWQ** fits a 6 GB laptop GPU. With 12 GB or more, you can use `Qwen/Qwen2.5-7B-Instruct-AWQ`, the model in our Gemini comparison.
- **`--gpu-memory-utilization 0.6`** leaves part of the GPU free for Windows.
- **`VLLM_USE_FLASHINFER_SAMPLER=0`** turns off one optional speed-up that needs the full CUDA toolkit (`nvcc`).
- **`--generation-config vllm`** makes the server use our settings, not the model's own defaults.

## 5. Start the MathEasy page

In a **second** PowerShell window, in the MathEasy folder, run:
```
python -m matheasy.ui --vllm --vllm-model Qwen/Qwen2.5-3B-Instruct-AWQ
```

Open **http://localhost:8001**. The badges should read **LLM off · vLLM on**. Hover over "vLLM on" to see the model. No Gemini key is needed.

For the Gemini version instead, run `python -m matheasy.ui --llm`. The badges then read LLM on · vLLM off.

### With Eedi on as well (a 6 GB GPU)

The Eedi shortlister is a second model on the same GPU, and Windows needs RAM to load it. To make both fit:

1. Cap WSL's RAM. Run `notepad $env:USERPROFILE\.wslconfig`, paste the lines below, and save:
   ```
   [wsl2]
   memory=5GB
   swap=0

   [experimental]
   sparseVhd=true
   ```
   - `memory=5GB` leaves Windows enough RAM for the Eedi model.
   - `swap=0` keeps WSL's swap file off C:.
   - `sparseVhd=true` gives space freed in Ubuntu back to Windows.

   Then, in PowerShell:
   ```
   wsl --shutdown
   wsl --manage Ubuntu --set-sparse true --allow-unsafe
   wsl -d Ubuntu -u root -- fstrim -v /
   ```
   `--allow-unsafe` is only a confirmation that WSL asks for.

2. Start vLLM with **`--gpu-memory-utilization 0.55`** instead of 0.6. 0.5 is too little for 4096-token requests.

3. Once the server is ready, start the page with Eedi:
   ```
   python -m matheasy.ui --vllm --vllm-model Qwen/Qwen2.5-3B-Instruct-AWQ --eedi
   ```
   Loading the Eedi model takes a few minutes. The badges then read **LLM off · vLLM on · Eedi on**. The GPU is then almost full (about 5.9 of 6.1 GB). If an Analyze fails with "CUDA out of memory", use Gemini for Eedi demos instead (`--llm --eedi`).

## 6. Stop it

- Press **Ctrl+C** in each of the two windows.
- To free all of WSL's memory as well, run `wsl --shutdown` in PowerShell.

After a restart, repeat only **steps 4 and 5**.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| `Forbidden (403)` when installing WSL | Add `--web-download`; see step 1. |
| The download or output froze, and the title bar says **"Select"** | You clicked inside the window, which pauses it. Press **Esc**, and don't click inside while it's working. |
| Very slow downloads, or `incomplete-download` | Run the same `pip install` command again: finished files are kept. A phone hotspot can help. |
| `No space left on device` | Use the `TMPDIR=~/tmp` command from step 3, and free space on C:. `pip cache purge` (in Windows and in Ubuntu) can free several GB. |
| `Failed to find C compiler` | `sudo apt install -y build-essential` |
| `Could not find nvcc` | Start the server with `VLLM_USE_FLASHINFER_SAMPLER=0`, as in step 4. |
| `The paging file is too small` when loading Eedi | Windows is short of RAM and disk space. Cap WSL at 5 GB (see "With Eedi on as well") and free space on C:. |
| `0.14 GiB KV cache is needed, which is larger than the available` | Raise `--gpu-memory-utilization` a little (0.55), or lower `--max-model-len`. |
| `CUDA out of memory` | Use the 3B model, lower `--max-model-len` to 2048, and don't run `--eedi` at the same time on a 6 GB GPU. |
| The page says "no vLLM server answers at http://localhost:8000/v1" | The server isn't ready yet (wait for `Application startup complete`), or it stopped; start step 4 again. |
| The page says "the vLLM server serves [...], not ..." | Pass the model the server is running: `--vllm-model <that name>`. |

**Disk space:** on C:, WSL and Ubuntu take about 1.5 GB, vLLM about 9 GB, and each 3B model about 2 GB. To remove all of it, run `wsl --unregister Ubuntu` in PowerShell. This **deletes** that Ubuntu and everything in it.
