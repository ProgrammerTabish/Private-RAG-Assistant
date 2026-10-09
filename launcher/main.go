// install_run_rag.exe - Windows launcher for the Private RAG Assistant.
//
// Does the same as install_run_rag.sh, without admin rights:
//  1. restores the prebuilt embeddings + vector DB from rag_db\rag_db_data.tar.gz into db_data\
//  2. installs uv, Python 3.11 and the Python packages into .runtime\ (CPU or CUDA PyTorch)
//  3. installs Ollama into .runtime\ollama (or reuses a running/installed Ollama)
//  4. pulls mistral-small3.1:24b (weights in db_data\ollama_models) and keeps it loaded
//  5. starts the RAG API and prints the endpoint + API key
//
// Put install_run_rag.exe in the private-rag-assistant folder and double-click it.
// Flags: --host 0.0.0.0 --port 8000 --api-key KEY --model NAME --ollama-port 11434
package main

import (
	"archive/tar"
	"archive/zip"
	"bufio"
	"compress/gzip"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

var (
	flagHost       = flag.String("host", envOr("API_HOST", "0.0.0.0"), "address the API listens on (0.0.0.0 = whole network, 127.0.0.1 = this PC only)")
	flagPort       = flag.String("port", envOr("API_PORT", "8000"), "API port")
	flagKey        = flag.String("api-key", os.Getenv("PRIVRAG_API_KEY"), "API key clients must send as X-API-Key (default: generated once, saved in .runtime\\api_key.txt)")
	flagModel      = flag.String("model", envOr("LLM_MODEL", "mistral-small3.1:24b"), "Ollama model")
	flagOllamaPort = flag.String("ollama-port", envOr("OLLAMA_PORT", "11434"), "Ollama port (local only)")
	flagCtx        = flag.String("context", envOr("OLLAMA_CONTEXT_LENGTH", "16384"), "Ollama context length")
	flagNoPause    = flag.Bool("no-pause", false, "do not wait for Enter before closing the window")
	flagNoBrowser  = flag.Bool("no-browser", false, "do not open the chat UI in the browser")
)

var (
	root, runtimeDir, dbDir string
	children                []*exec.Cmd
)

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func logf(format string, a ...any) { fmt.Printf("[install_run_rag] "+format+"\n", a...) }

func fail(format string, a ...any) {
	fmt.Printf("\n[install_run_rag] ERROR: "+format+"\n", a...)
	stopChildren()
	pause()
	os.Exit(1)
}

func pause() {
	if *flagNoPause {
		return
	}
	fmt.Print("\nPress Enter to close this window ...")
	bufio.NewReader(os.Stdin).ReadString('\n')
}

func exists(p string) bool { _, err := os.Stat(p); return err == nil }

// ------------------------------------------------------------------ main
func main() {
	flag.Parse()
	findRoot()
	runtimeDir = filepath.Join(root, ".runtime")
	dbDir = envOr("DB_DIR", filepath.Join(root, "db_data"))
	must(os.MkdirAll(runtimeDir, 0o755))

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sig
		fmt.Println("\n[install_run_rag] stopping ...")
		stopChildren()
		os.Exit(130)
	}()

	logf("project folder: %s", root)
	gpu := detectGPU()
	restoreDB()
	python, privrag := setupPython(gpu)
	ollamaURL := setupOllama()
	pullAndLoad(ollamaURL)
	key := apiKey()
	runAPI(python, privrag, ollamaURL, key, gpu)
}

func must(err error) {
	if err != nil {
		fail("%v", err)
	}
}

func findRoot() {
	cands := []string{}
	if exe, err := os.Executable(); err == nil {
		cands = append(cands, filepath.Dir(exe))
	}
	if wd, err := os.Getwd(); err == nil {
		cands = append(cands, wd)
	}
	for _, c := range cands {
		if exists(filepath.Join(c, "selfhosted", "pyproject.toml")) {
			root = c
			return
		}
	}
	fail("put install_run_rag.exe into the private-rag-assistant folder (next to selfhosted\\ and rag_db\\)")
}

// ------------------------------------------------------------------ 1. GPU
func detectGPU() bool {
	out, err := exec.Command("nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader").Output()
	if err == nil && strings.TrimSpace(string(out)) != "" {
		logf("NVIDIA GPU(s):")
		for _, l := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			fmt.Println("    " + strings.TrimSpace(l))
		}
		return true
	}
	logf("no NVIDIA GPU - running on CPU (prebuilt embeddings are used; Mistral 24B needs ~20 GB RAM, answers take minutes)")
	return false
}

// ------------------------------------------------------------------ 2. vector DB
func restoreDB() {
	meta := filepath.Join(dbDir, "index_meta.path.spg_compliance.json")
	if exists(meta) && exists(filepath.Join(dbDir, "qdrant")) {
		logf("vector DB: %s", dbDir)
		return
	}
	archive := filepath.Join(root, "rag_db", "rag_db_data.tar.gz")
	if !exists(archive) {
		fail("no vector database in %s and no prebuilt archive at %s", dbDir, archive)
	}
	if b, err := os.ReadFile(archive + ".sha256"); err == nil {
		want := strings.Fields(string(b))[0]
		logf("verifying %s ...", filepath.Base(archive))
		have := fileSHA256(archive)
		if !strings.EqualFold(want, have) {
			fail("checksum mismatch for %s (expected %s, got %s)", archive, want, have)
		}
	}
	logf("unpacking prebuilt embeddings + vector DB into %s ...", dbDir)
	tmp := filepath.Join(runtimeDir, "rag_db_unpack")
	os.RemoveAll(tmp)
	must(untarGz(archive, tmp))
	src := filepath.Join(tmp, "db_data")
	if !exists(src) {
		fail("archive does not contain a db_data folder")
	}
	if exists(dbDir) {
		must(os.Rename(dbDir, fmt.Sprintf("%s.old.%d", dbDir, time.Now().Unix())))
	}
	must(os.MkdirAll(filepath.Dir(dbDir), 0o755))
	must(os.Rename(src, dbDir))
	os.RemoveAll(tmp)
	if !exists(meta) {
		fail("vector database in %s is incomplete", dbDir)
	}
}

func fileSHA256(p string) string {
	f, err := os.Open(p)
	must(err)
	defer f.Close()
	h := sha256.New()
	_, err = io.Copy(h, f)
	must(err)
	return hex.EncodeToString(h.Sum(nil))
}

func safeJoin(dest, name string) (string, error) {
	p := filepath.Join(dest, filepath.FromSlash(name))
	if !strings.HasPrefix(p, filepath.Clean(dest)+string(os.PathSeparator)) {
		return "", fmt.Errorf("unsafe path in archive: %s", name)
	}
	return p, nil
}

func untarGz(archive, dest string) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		p, err := safeJoin(dest, h.Name)
		if err != nil {
			return err
		}
		switch h.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				return err
			}
			out, err := os.Create(p)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			out.Close()
		}
	}
}

func unzip(archive, dest string) error {
	r, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer r.Close()
	for _, f := range r.File {
		p, err := safeJoin(dest, f.Name)
		if err != nil {
			return err
		}
		if f.FileInfo().IsDir() {
			os.MkdirAll(p, 0o755)
			continue
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.Create(p)
		if err != nil {
			rc.Close()
			return err
		}
		_, err = io.Copy(out, rc)
		rc.Close()
		out.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

// download with a progress line every 5 %
func download(url, dest string) error {
	logf("downloading %s", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("%s -> HTTP %d", url, resp.StatusCode)
	}
	tmp := dest + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	total := resp.ContentLength
	var done int64
	lastPct := -5
	buf := make([]byte, 1<<20)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := out.Write(buf[:n]); err != nil {
				out.Close()
				return err
			}
			done += int64(n)
			if total > 0 {
				if pct := int(done * 100 / total); pct >= lastPct+5 {
					fmt.Printf("    %3d%%  %d / %d MB\n", pct, done>>20, total>>20)
					lastPct = pct
				}
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			out.Close()
			return rerr
		}
	}
	out.Close()
	return os.Rename(tmp, dest)
}

// ------------------------------------------------------------------ 3. Python
func run(name string, args []string, env []string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdout, cmd.Stderr, cmd.Stdin = os.Stdout, os.Stderr, nil
	cmd.Env = append(os.Environ(), env...)
	return cmd.Run()
}

func setupPython(gpu bool) (python, privrag string) {
	uvDir := filepath.Join(runtimeDir, "uv")
	uv := filepath.Join(uvDir, "uv.exe")
	if runtime.GOOS != "windows" {
		uv = filepath.Join(uvDir, "uv")
	}
	if !exists(uv) {
		logf("installing uv (Python package manager) ...")
		must(os.MkdirAll(uvDir, 0o755))
		zipPath := filepath.Join(runtimeDir, "uv.zip")
		must(download("https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip", zipPath))
		must(unzip(zipPath, uvDir))
		os.Remove(zipPath)
		if !exists(uv) { // some releases put the binaries in a sub folder
			filepath.Walk(uvDir, func(p string, i os.FileInfo, _ error) error {
				if i != nil && !i.IsDir() && strings.EqualFold(i.Name(), "uv.exe") && p != uv {
					os.Rename(p, uv)
				}
				return nil
			})
		}
		if !exists(uv) {
			fail("uv.exe not found after download")
		}
	}
	uvEnv := []string{
		"UV_CACHE_DIR=" + filepath.Join(runtimeDir, "uv-cache"),
		"UV_PYTHON_INSTALL_DIR=" + filepath.Join(runtimeDir, "python"),
		"UV_LINK_MODE=copy",
	}
	venv := filepath.Join(runtimeDir, "venv")
	python = filepath.Join(venv, "Scripts", "python.exe")
	privrag = filepath.Join(venv, "Scripts", "privrag.exe")
	if !exists(python) {
		logf("creating Python 3.11 environment (downloaded automatically) ...")
		if err := run(uv, []string{"venv", "--python", "3.11", venv}, uvEnv); err != nil {
			fail("creating the Python environment failed: %v", err)
		}
	}
	torchIndex := envOr("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu")
	if gpu && os.Getenv("TORCH_INDEX_URL") == "" {
		torchIndex = "https://download.pytorch.org/whl/cu124"
	}
	logf("installing Python packages (PyTorch from %s, sentence-transformers, privrag) ...", torchIndex)
	if err := run(uv, []string{"pip", "install", "--python", python, "--extra-index-url", torchIndex, "torch>=2.2"}, uvEnv); err != nil {
		fail("installing PyTorch failed: %v", err)
	}
	if err := run(uv, []string{"pip", "install", "--python", python, "-e", filepath.Join(root, "selfhosted"),
		"torch>=2.2", "sentence-transformers>=3.0"}, uvEnv); err != nil {
		fail("installing the Python packages failed: %v", err)
	}
	return python, privrag
}

// ------------------------------------------------------------------ 4. Ollama
func ollamaUp(url string) bool {
	c := http.Client{Timeout: 2 * time.Second}
	r, err := c.Get(url + "/api/version")
	if err != nil {
		return false
	}
	r.Body.Close()
	return r.StatusCode == 200
}

func findOllamaExe() string {
	dir := filepath.Join(runtimeDir, "ollama")
	found := ""
	filepath.Walk(dir, func(p string, i os.FileInfo, _ error) error {
		if found == "" && i != nil && !i.IsDir() && strings.EqualFold(i.Name(), "ollama.exe") {
			found = p
		}
		return nil
	})
	return found
}

func setupOllama() string {
	url := "http://127.0.0.1:" + *flagOllamaPort
	if ollamaUp(url) {
		logf("an Ollama server is already running on %s - using it", url)
		return url
	}
	exe := findOllamaExe()
	if exe == "" {
		if p, err := exec.LookPath("ollama"); err == nil {
			exe = p
			logf("using installed Ollama: %s", exe)
		}
	}
	if exe == "" {
		dir := filepath.Join(runtimeDir, "ollama")
		must(os.MkdirAll(dir, 0o755))
		zipPath := filepath.Join(runtimeDir, "ollama-windows-amd64.zip")
		if err := download("https://ollama.com/download/ollama-windows-amd64.zip", zipPath); err != nil {
			logf("ollama.com download failed (%v) - trying GitHub", err)
			must(download("https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip", zipPath))
		}
		logf("unpacking Ollama ...")
		must(unzip(zipPath, dir))
		os.Remove(zipPath)
		if exe = findOllamaExe(); exe == "" {
			fail("ollama.exe not found after download")
		}
	}
	must(os.MkdirAll(filepath.Join(dbDir, "logs"), 0o755))
	must(os.MkdirAll(filepath.Join(dbDir, "ollama_models"), 0o755))
	logFile, err := os.OpenFile(filepath.Join(dbDir, "logs", "ollama.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	must(err)
	cmd := exec.Command(exe, "serve")
	cmd.Stdout, cmd.Stderr = logFile, logFile
	cmd.Env = append(os.Environ(), ollamaEnv()...)
	logf("starting Ollama on %s (log: db_data\\logs\\ollama.log) ...", url)
	must(cmd.Start())
	children = append(children, cmd)
	for i := 0; i < 60 && !ollamaUp(url); i++ {
		time.Sleep(time.Second)
	}
	if !ollamaUp(url) {
		fail("Ollama did not start - see %s", filepath.Join(dbDir, "logs", "ollama.log"))
	}
	ollamaExe = exe
	return url
}

var ollamaExe string

func ollamaEnv() []string {
	return []string{
		"OLLAMA_HOST=127.0.0.1:" + *flagOllamaPort,
		"OLLAMA_MODELS=" + filepath.Join(dbDir, "ollama_models"),
		"OLLAMA_CONTEXT_LENGTH=" + *flagCtx,
		"OLLAMA_KEEP_ALIVE=-1",
	}
}

func pullAndLoad(url string) {
	logf("pulling %s (~15 GB the first time, then cached) ...", *flagModel)
	if ollamaExe != "" {
		if err := run(ollamaExe, []string{"pull", *flagModel}, ollamaEnv()); err != nil {
			fail("ollama pull %s failed: %v", *flagModel, err)
		}
	} else { // reused server whose binary we don't know: pull over the HTTP API
		body, _ := json.Marshal(map[string]any{"model": *flagModel, "stream": false})
		c := http.Client{Timeout: 4 * time.Hour}
		r, err := c.Post(url+"/api/pull", "application/json", strings.NewReader(string(body)))
		if err != nil {
			fail("pulling %s failed: %v", *flagModel, err)
		}
		io.Copy(io.Discard, r.Body)
		r.Body.Close()
		if r.StatusCode != 200 {
			fail("pulling %s failed: HTTP %d", *flagModel, r.StatusCode)
		}
	}
	logf("loading %s into memory (can take a few minutes on CPU) ...", *flagModel)
	body, _ := json.Marshal(map[string]any{"model": *flagModel, "prompt": "Antworte nur mit OK.", "stream": false, "keep_alive": -1})
	c := http.Client{Timeout: 30 * time.Minute}
	r, err := c.Post(url+"/api/generate", "application/json", strings.NewReader(string(body)))
	if err != nil {
		fail("loading %s failed: %v", *flagModel, err)
	}
	io.Copy(io.Discard, r.Body)
	r.Body.Close()
}

// ------------------------------------------------------------------ 5. API
func apiKey() string {
	if *flagKey != "" {
		return *flagKey
	}
	p := filepath.Join(runtimeDir, "api_key.txt")
	if b, err := os.ReadFile(p); err == nil && strings.TrimSpace(string(b)) != "" {
		return strings.TrimSpace(string(b))
	}
	b := make([]byte, 24)
	rand.Read(b)
	k := hex.EncodeToString(b)
	must(os.WriteFile(p, []byte(k+"\n"), 0o600))
	return k
}

func localIPs() []string {
	var ips []string
	addrs, _ := net.InterfaceAddrs()
	for _, a := range addrs {
		if n, ok := a.(*net.IPNet); ok && !n.IP.IsLoopback() && n.IP.To4() != nil && !n.IP.IsLinkLocalUnicast() {
			ips = append(ips, n.IP.String())
		}
	}
	return ips
}

func runAPI(python, privrag, ollamaURL, key string, gpu bool) {
	hf := filepath.Join(dbDir, "hf_cache")
	timeout := "1800"
	if gpu {
		timeout = "300"
	}
	env := []string{
		"PYTHONUTF8=1", "PYTHONIOENCODING=utf-8",
		"HF_HOME=" + hf, "HF_HUB_DISABLE_SYMLINKS_WARNING=1",
		"PRIVRAG_ENV=local",
		"PRIVRAG_DATA_DIR=" + dbDir,
		"PRIVRAG_LOG_DIR=" + filepath.Join(dbDir, "logs"),
		"PRIVRAG_PDF_DIR=" + filepath.Join(root, "documents", "spg_compliance"),
		"PRIVRAG_MANIFEST_PATH=" + filepath.Join(root, "documents", "manifest.csv"),
		"PRIVRAG_EVAL_PATH=" + filepath.Join(root, "documents", "UC13_Evaluation_Question_Set_Students.xlsx"),
		"PRIVRAG_QDRANT_MODE=path",
		"PRIVRAG_EMBED_BACKEND=bge-m3",
		"PRIVRAG_EMBED_MODEL=BAAI/bge-m3",
		"PRIVRAG_EMBED_DEVICE=auto",
		"PRIVRAG_LLM_BACKEND=openai",
		"PRIVRAG_LLM_BASE_URL=" + ollamaURL + "/v1",
		"PRIVRAG_LLM_MODEL=" + *flagModel,
		"PRIVRAG_LLM_TIMEOUT_S=" + envOr("LLM_TIMEOUT_S", timeout),
		"PRIVRAG_LLM_MAX_RETRIES=1",
		"PRIVRAG_API_KEY=" + key,
	}
	if exists(filepath.Join(hf, "hub", "models--BAAI--bge-m3")) {
		env = append(env, "HF_HUB_OFFLINE=1")
	}
	if !exists(privrag) {
		fail("privrag not installed (%s missing)", privrag)
	}
	logf("checking index + LLM endpoint ...")
	if err := run(privrag, []string{"doctor"}, env); err != nil {
		fail("privrag doctor failed - see the output above")
	}

	cmd := exec.Command(privrag, "serve", "--host", *flagHost, "--port", *flagPort)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	cmd.Env = append(os.Environ(), env...)
	must(cmd.Start())
	children = append(children, cmd)

	// wait until /health answers, then print the endpoint
	go func() {
		c := http.Client{Timeout: 3 * time.Second}
		for i := 0; i < 600; i++ {
			if r, err := c.Get("http://127.0.0.1:" + *flagPort + "/health"); err == nil {
				r.Body.Close()
				banner(key)
				if !*flagNoBrowser {
					openBrowser(fmt.Sprintf("http://127.0.0.1:%s/#key=%s", *flagPort, key))
				}
				return
			}
			time.Sleep(2 * time.Second)
		}
	}()
	err := cmd.Wait()
	stopChildren()
	if err != nil && !errors.Is(err, os.ErrProcessDone) {
		fail("the API stopped: %v", err)
	}
	pause()
}

func banner(key string) {
	line := strings.Repeat("=", 78)
	fmt.Println("\n" + line)
	fmt.Println(" RAG assistant is running   (close this window or press Ctrl+C to stop)")
	fmt.Println(line)
	fmt.Printf(" Chat UI:        http://127.0.0.1:%s/   (opens automatically)\n", *flagPort)
	fmt.Printf(" API (this PC):  http://127.0.0.1:%s/ask\n", *flagPort)
	if *flagHost == "0.0.0.0" {
		for _, ip := range localIPs() {
			fmt.Printf(" Network:        http://%s:%s/   (UI)   http://%s:%s/ask   (API)\n", ip, *flagPort, ip, *flagPort)
		}
	} else if *flagHost != "127.0.0.1" {
		fmt.Printf(" Network:        http://%s:%s/ask\n", *flagHost, *flagPort)
	}
	fmt.Printf(" API key:        %s   (header X-API-Key)\n", key)
	fmt.Printf(" API docs:       http://127.0.0.1:%s/docs\n", *flagPort)
	fmt.Println(" Example (PowerShell):")
	fmt.Printf("   curl.exe -X POST http://127.0.0.1:%s/ask -H \"Content-Type: application/json\" -H \"X-API-Key: %s\" `\n", *flagPort, key)
	fmt.Println("     -d '{\\\"question\\\": \\\"Welche Meldefristen gelten nach DORA?\\\"}'")
	if *flagHost == "0.0.0.0" {
		fmt.Println(" Other machines need Windows Firewall to allow TCP port " + *flagPort + " (allow the prompt, or as admin:")
		fmt.Println("   netsh advfirewall firewall add rule name=privrag dir=in action=allow protocol=TCP localport=" + *flagPort + ")")
	}
	fmt.Println(line + "\n")
}

func stopChildren() {
	for i := len(children) - 1; i >= 0; i-- {
		if p := children[i].Process; p != nil {
			p.Kill()
		}
	}
	children = nil
}

// openBrowser opens the chat UI; the API key travels in the URL fragment, which is never sent to the server.
func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	if err := cmd.Start(); err != nil {
		logf("could not open the browser (%v) - open http://127.0.0.1:%s/ yourself", err, *flagPort)
	}
}
