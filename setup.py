"""
setup.py — Instalador do Samuel V2 (Windows e Linux).

Uso:
    python setup.py            # instala tudo: dependências, pacotes do sistema e modelos
    python setup.py --vad      # só (re)instala o VAD neural (silero_vad.onnx)
    python setup.py --modelos  # só baixa os modelos (Vosk, Silero VAD, voz do Piper)

O que ele faz:
  1. Detecta o sistema operacional (Windows/Linux e o gerenciador de pacotes).
  2. Instala pacotes de sistema necessários (portaudio, ffmpeg, piper-bin...).
  3. Cria/atualiza o venv (opcional, se rodar dentro de um já ativa ele usa o atual).
  4. Instala as dependências Python do requirements.txt.
  5. Baixa os modelos: Vosk (wakeword), Silero VAD (ONNX) e a voz pt_BR do Piper.
"""

import argparse
import os
import platform
import subprocess
import sys
import urllib.request
import zipfile

AQUI = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"

# ---------------- RECURSOS A BAIXAR ----------------
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip"
VOSK_PASTA = "vosk-model-small-pt-0.3"

SILERO_URLS = [
    # v5 (com estado) e v4 (h/c) — o main.py usa a que carregar
    ("https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
     "silero_vad.onnx"),
    ("https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad_v4.onnx",
     "silero_vad.onnx"),
]

PIPER_VOZ_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx"
PIPER_CONF_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json"
PIPER_VOZ = "pt_BR-faber-medium.onnx"


def _tem(nome: str) -> bool:
    from shutil import which
    return which(nome) is not None


def _rodar(cmd: list[str], root: bool = False) -> bool:
    """Roda um comando; se precisar de root e não for root, usa sudo quando existir."""
    eh_root = hasattr(os, "geteuid") and os.geteuid() == 0
    if root and not eh_root:
        if _tem("sudo"):
            cmd = ["sudo"] + cmd
    print(f"  $ {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd)
        return res.returncode == 0
    except FileNotFoundError:
        print(f"  ✗ '{cmd[0]}' não encontrado.")
        return False


def instalar_pacotes_sistema():
    """Instala dependências nativas (portaudio, ffmpeg) conforme o OS/gerenciador."""
    print("\n[1/4] Verificando pacotes de sistema...")

    if IS_WINDOWS:
        print("  Windows: os pacotes necessários (portaudio) já vêm via pip com sounddevice.")
        if not _tem("ffmpeg"):
            print("  ⚠ ffmpeg não encontrado (opcional). Instale com:")
            print("    winget install Gyan.FFmpeg   ou   choco install ffmpeg")
        return

    if IS_MAC:
        if _tem("brew"):
            _rodar(["brew", "install", "portaudio", "ffmpeg"])
        else:
            print("  ⚠ Homebrew não encontrado. Instale portaudio e ffmpeg manualmente.")
        return

    # Linux
    if _tem("pacman"):
        _rodar(["pacman", "-S", "--needed", "-y", "portaudio", "ffmpeg"], root=True)
    elif _tem("apt-get"):
        _rodar(["apt-get", "install", "-y", "portaudio19-dev", "ffmpeg", "python3-dev"], root=True)
    elif _tem("dnf"):
        _rodar(["dnf", "install", "-y", "portaudio-devel", "ffmpeg"])
    elif _tem("zypper"):
        _rodar(["zypper", "install", "-y", "portaudio-devel", "ffmpeg"])
    else:
        print("  ⚠ Gerenciador de pacotes não reconhecido. Instale portaudio e ffmpeg manualmente.")


def instalar_dependencias_python():
    """Instala requirements.txt no Python atual (de preferência num venv)."""
    print("\n[2/4] Instalando dependências Python (pip install -r requirements.txt)...")
    if IS_LINUX and hasattr(sys, "base_prefix") and sys.base_prefix == sys.prefix \
            and os.environ.get("VIRTUAL_ENV") is None:
        print("  ⚠ Você não está num venv. Recomendo rodar de dentro de um:")
        print("    python -m venv .venv && source .venv/bin/activate")

    # piper-tts precisa do sdist às vezes; forçar wheels pré-compilados:
    _rodar([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    ok = _rodar([sys.executable, "-m", "pip", "install", "-r",
                 os.path.join(AQUI, "requirements.txt")])
    if not ok:
        print("  ⚠ Falhou o pip. Tentando sem as dependências exclusivas de Windows...")
        base = ["numpy", "openai", "python-dotenv", "sounddevice", "faster-whisper",
                "piper-tts", "vosk", "onnxruntime", "pyperclip", "psutil"]
        _rodar([sys.executable, "-m", "pip", "install"] + base)


def _baixar(url: str, destino: str) -> bool:
    """Baixa uma URL para um arquivo, com barra de progresso simples."""
    if os.path.isfile(destino) and os.path.getsize(destino) > 0:
        print(f"  ✓ Já existe: {destino}")
        return True
    print(f"  ↓ Baixando {url}")
    os.makedirs(os.path.dirname(destino) or ".", exist_ok=True)
    try:
        urllib.request.urlretrieve(url, destino)
        print(f"  ✓ Salvo em: {destino} ({os.path.getsize(destino) // 1024} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Falhou: {e}")
        return False


def instalar_vad():
    """Baixa o silero_vad.onnx (V5, fallback V4)."""
    print("\n[3/4] Instalando VAD neural (Silero)...")
    for url, nome in SILERO_URLS:
        destino = os.path.join(AQUI, nome)
        if _baixar(url, destino):
            # valida: precisa ser binário grande, e idealmente abrir no onnxruntime
            if os.path.getsize(destino) < 100_000:
                os.remove(destino)
                print("  ✗ Arquivo pequeno demais (provável 404), tentando espelho seguinte...")
                continue
            try:
                import onnxruntime as ort
                ort.InferenceSession(destino, providers=["CPUExecutionProvider"])
            except ImportError:
                pass  # sem onnxruntime, aceita só pelo tamanho
            except Exception as e:
                os.remove(destino)
                print(f"  ✗ ONNX inválido ({e}), tentando espelho seguinte...")
                continue
            print("  ✓ VAD neural instalado.")
            return True
    print("  ⚠ Sem VAD neural — o Samuel usará VAD por energia (funciona, mas é pior).")
    return False


def instalar_modelos():
    """Baixa modelo Vosk, Silero VAD e a voz do Piper."""
    print("\n[4/4] Baixando modelos de voz...")

    # Vosk
    if os.path.isdir(os.path.join(AQUI, VOSK_PASTA)):
        print(f"  ✓ Vosk já presente: {VOSK_PASTA}/")
    else:
        zip_path = os.path.join(AQUI, VOSK_PASTA + ".zip")
        if _baixar(VOSK_URL, zip_path):
            print(f"  ... extraindo {VOSK_PASTA}.zip")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(AQUI)
            os.remove(zip_path)
            print(f"  ✓ Vosk extraído: {VOSK_PASTA}/")

    # VAD neural
    instalar_vad()

    # Voz Piper
    if not _baixar(PIPER_VOZ_URL, os.path.join(AQUI, PIPER_VOZ)):
        print(f"  ✗ Sem a voz do Piper! Baixe manualmente de:")
        print(f"    {PIPER_VOZ_URL}")
    else:
        _baixar(PIPER_CONF_URL, os.path.join(AQUI, PIPER_VOZ + ".json"))


def verificar():
    """Confere se tudo que o main.py precisa está no lugar."""
    print("\n" + "=" * 55)
    print("Verificação final:")
    ok = True

    # imports críticos
    for modulo in ("numpy", "openai", "sounddevice", "faster_whisper", "piper",
                   "vosk", "onnxruntime", "psutil", "pyperclip"):
        try:
            __import__(modulo)
            print(f"  ✓ Python: {modulo}")
        except ImportError:
            print(f"  ✗ Python: {modulo} — rode 'pip install -r requirements.txt'")
            ok = False

    for arquivo in (VOSK_PASTA, "silero_vad.onnx", PIPER_VOZ):
        if os.path.exists(os.path.join(AQUI, arquivo)):
            print(f"  ✓ Modelo: {arquivo}")
        else:
            print(f"  ✗ Modelo: {arquivo} — rode 'python setup.py --modelos'")
            ok = False

    print("=" * 55)
    if ok:
        print("🟢 Tudo pronto! Rode:  python main.py")
        print("   Lembre de definir OPENROUTER_API_KEY no .env")
    else:
        print("🟡 Faltam itens acima — rode de novo o setup ou o passo indicado.")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Instalador do Samuel V2")
    parser.add_argument("--vad", action="store_true", help="só instala o silero_vad.onnx")
    parser.add_argument("--modelos", action="store_true", help="só baixa os modelos")
    parser.add_argument("--sem-sistema", action="store_true",
                        help="pula a instalação de pacotes do sistema")
    args = parser.parse_args()

    print("🤖 SETUP DO SAMUEL V2")
    print(f"   SO detectado: {platform.system()} ({platform.release()})")

    if args.vad:
        instalar_vad()
        return
    if args.modelos:
        instalar_modelos()
        return

    if not args.sem_sistema:
        instalar_pacotes_sistema()
    instalar_dependencias_python()
    instalar_modelos()
    verificar()


if __name__ == "__main__":
    main()
