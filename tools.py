"""
tools.py — Ferramentas que o Samuel V2 pode usar (chamadas pelo LLM).

Cada função pública (sem "_" no começo) desta arquivo é automaticamente
registrada como ferramenta do LLM pelo main.py, usando o type hints + docstring.
O arquivo detecta o sistema operacional (Windows/Linux/macOS) e usa os
comandos/utilitários corretos para cada um.
"""

import os
import platform
import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime

# ---------------- DETECÇÃO DE SISTEMA OPERACIONAL ----------------
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"
OS_NAME = platform.system()  # "Windows", "Linux" ou "Darwin"


def _which(*nomes: str) -> str | None:
    """Retorna o primeiro executável disponível dentre os nomes dados."""
    for nome in nomes:
        caminho = shutil.which(nome)
        if caminho:
            return caminho
    return None


def _run(cmd: list[str], timeout: float = 15.0) -> str:
    """Executa um comando e retorna stdout (ou stderr em caso de erro)."""
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, errors="replace"
        )
        saida = (res.stdout or "").strip()
        if not saida and res.returncode != 0:
            saida = (res.stderr or "").strip() or f"código de saída {res.returncode}"
        return saida
    except FileNotFoundError:
        return f"Erro: comando '{cmd[0]}' não encontrado neste sistema."
    except subprocess.TimeoutExpired:
        return f"Erro: comando '{cmd[0]}' demorou demais ({timeout}s)."
    except Exception as e:
        return f"Erro ao executar '{cmd[0]}': {e}"


# ---------------- INFORMAÇÕES DO SISTEMA ----------------
def sistema_info() -> str:
    """Retorna informações do sistema operacional: nome, versão, CPU, memória e bateria.

    Args:
        (nenhum)
    """
    linhas = [
        f"Sistema: {OS_NAME}",
        f"Release: {platform.release()}",
        f"Arquitetura: {platform.machine()}",
        f"Python: {sys.version.split()[0]}",
    ]
    if IS_LINUX:
        linhas.append(f"Distro: {' '.join(platform.freedesktop_os_release().get('PRETTY_NAME', '').split()) or 'desconhecida'}")
    elif IS_WINDOWS:
        linhas.append(f"Versão Windows: {platform.win32_ver()}")

    # CPU / memória via psutil (opcional)
    try:
        import psutil
        linhas.append(f"CPU: {psutil.cpu_percent(interval=0.3)}% em uso ({psutil.cpu_count()} núcleos)")
        mem = psutil.virtual_memory()
        linhas.append(f"RAM: {mem.percent}% usada ({mem.used // (1024**3)} GB de {mem.total // (1024**3)} GB)")
        bateria = psutil.sensors_battery()
        if bateria:
            status = "carregando" if bateria.power_plugged else "na bateria"
            linhas.append(f"Bateria: {round(bateria.percent)}% ({status})")
    except ImportError:
        linhas.append("(psutil não instalado — sem CPU/RAM/bateria)")
    except Exception as e:
        linhas.append(f"(psutil falhou: {e})")

    return "\n".join(linhas)


def data_hora() -> str:
    """Retorna a data e hora atuais do computador, em português.

    Args:
        (nenhum)
    """
    agora = datetime.now()
    dias = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
    meses = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
             "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
    return (
        f"Agora é {agora.strftime('%H:%M')} de {dias[agora.weekday()]}, "
        f"{agora.day} de {meses[agora.month - 1]} de {agora.year}."
    )


# ---------------- ABRIR COISAS ----------------
def abrir_site(url: str) -> str:
    """Abre uma URL/site no navegador padrão do usuário.

    Args:
        url: A URL completa ou domínio do site (ex.: "youtube.com").
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        webbrowser.open(url)
        return f"Abrindo {url} no navegador."
    except Exception as e:
        return f"Erro ao abrir o site: {e}"


def abrir_arquivo_ou_app(alvo: str) -> str:
    """Abre um aplicativo, arquivo ou pasta com o programa padrão do sistema.

    Args:
        alvo: Nome do app/executável ou caminho de arquivo/pasta (ex.: "code", "C:/Users/eu/doc.pdf", "~/Documentos").
    """
    alvo = os.path.expanduser(alvo)

    # É um caminho que existe?
    if os.path.exists(alvo):
        try:
            if IS_WINDOWS:
                os.startfile(alvo)  # type: ignore[attr-defined]
            elif IS_MAC:
                subprocess.Popen(["open", alvo])
            else:
                subprocess.Popen(["xdg-open", alvo])
            return f"Abrindo '{alvo}'."
        except Exception as e:
            return f"Erro ao abrir '{alvo}': {e}"

    # É um executável no PATH?
    caminho = _which(alvo, alvo + (".exe" if IS_WINDOWS else ""))
    if caminho:
        try:
            if IS_WINDOWS:
                subprocess.Popen([caminho], shell=False)
            else:
                subprocess.Popen([caminho], start_new_session=True)
            return f"Abrindo o aplicativo '{alvo}'."
        except Exception as e:
            return f"Erro ao abrir o app '{alvo}': {e}"

    return f"Não encontrei '{alvo}' nem como caminho nem como aplicativo instalado."


# ---------------- ARQUIVOS ----------------
def listar_arquivos(caminho: str = "~") -> str:
    """Lista os arquivos e pastas de um diretório.

    Args:
        caminho: Caminho do diretório (padrão: pasta pessoal do usuário). Aceita "~".
    """
    caminho = os.path.expanduser(caminho)
    if not os.path.isdir(caminho):
        return f"Erro: '{caminho}' não é uma pasta válida."
    try:
        itens = sorted(os.listdir(caminho))
        if not itens:
            return f"A pasta '{caminho}' está vazia."
        pastas = [i for i in itens if os.path.isdir(os.path.join(caminho, i))]
        arquivos = [i for i in itens if i not in pastas]
        partes = []
        if pastas:
            partes.append("Pastas: " + ", ".join(pastas[:30]))
        if arquivos:
            partes.append("Arquivos: " + ", ".join(arquivos[:40]))
        return f"Conteúdo de '{caminho}':\n" + "\n".join(partes)
    except Exception as e:
        return f"Erro ao listar '{caminho}': {e}"


def ler_arquivo(caminho: str, max_caracteres: int = 3000) -> str:
    """Lê um arquivo de texto e retorna seu conteúdo (truncado se for muito grande).

    Args:
        caminho: Caminho do arquivo a ser lido.
        max_caracteres: Número máximo de caracteres a retornar (padrão 3000).
    """
    caminho = os.path.expanduser(caminho)
    if not os.path.isfile(caminho):
        return f"Erro: '{caminho}' não existe."
    try:
        with open(caminho, "r", encoding="utf-8", errors="replace") as f:
            conteudo = f.read(max_caracteres + 1)
        if len(conteudo) > max_caracteres:
            return conteudo[:max_caracteres] + "\n[...truncado]"
        return conteudo or "(arquivo vazio)"
    except Exception as e:
        return f"Erro ao ler '{caminho}': {e}"


def escrever_arquivo(caminho: str, conteudo: str, acrescentar: bool = False) -> str:
    """Cria, sobrescreve ou acrescenta texto a um arquivo.

    Args:
        caminho: Caminho do arquivo (criado se não existir).
        conteudo: O texto completo a ser gravado.
        acrescentar: Se true, adiciona ao final sem apagar o que já existe (padrão: false).
    """
    caminho = os.path.expanduser(caminho)
    try:
        os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
        existia = os.path.isfile(caminho)
        modo = "a" if acrescentar else "w"
        with open(caminho, modo, encoding="utf-8") as f:
            f.write(conteudo)
        acao = "atualizado" if (existia and not acrescentar) else ("acrescentado" if acrescentar else "criado")
        return f"Arquivo '{caminho}' {acao} ({len(conteudo)} caracteres)."
    except Exception as e:
        return f"Erro ao escrever '{caminho}': {e}"


# ---------------- PROCESSOS ----------------
def processos_ativos(filtro: str = "") -> str:
    """Lista os processos em execução, opcionalmente filtrados por nome (útil para achar apps que travaram).

    Args:
        filtro: Se informado, mostra só processos cujo nome contém esse texto (padrão: todos os principais).
    """
    try:
        import psutil
    except ImportError:
        # fallback simples por OS
        if IS_WINDOWS:
            return _run(["tasklist"], timeout=20)
        return _run(["ps", "-eo", "pid,comm", "--sort=-pcpu"], timeout=10)

    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        info = p.info
        nome = info["name"] or "?"
        if filtro and filtro.lower() not in nome.lower():
            continue
        procs.append(f"{info['pid']:>7}  {nome[:40]:<40} CPU {info['cpu_percent'] or 0:.1f}%  RAM {info['memory_percent'] or 0:.1f}%")
        if len(procs) >= 40:
            break
    if not procs:
        return f"Nenhum processo encontrado com filtro '{filtro}'."
    return "\n".join(procs)


def encerrar_processo(nome: str) -> str:
    """Encerra (mata) um processo pelo nome, ex.: um programa travado.

    Args:
        nome: Nome do processo/executável (ex.: "firefox", "vlc", "notepad.exe").
    """
    if IS_WINDOWS:
        saida = _run(["taskkill", "/F", "/IM", nome if nome.endswith(".exe") else nome + ".exe"])
    else:
        saida = _run(["pkill", "-f", nome])
    if "Erro" in saida:
        return saida
    return f"Comando enviado para encerrar '{nome}': {saida}"


# ---------------- MULTIMÍDIA / DESKTOP ----------------
def controlar_volume(acao: str, nivel: int = 50) -> str:
    """Controla o volume do sistema: aumentar, diminuir, definir nível ou silenciar.

    Args:
        acao: "aumentar", "diminuir", "definir", "mudo" ou "status".
        nivel: Percentual 0-100 (usado apenas com a ação "definir").
    """
    acao = acao.lower()

    if IS_WINDOWS:
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            dispositivos = AudioUtilities.GetSpeakers()
            interface = dispositivos.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol = cast(interface, POINTER(IAudioEndpointVolume))
            if acao == "aumentar":
                atual = vol.GetMasterVolumeLevelScalar() * 100
                vol.SetMasterVolumeLevelScalar(min(1.0, (atual + 10) / 100), None)
            elif acao == "diminuir":
                atual = vol.GetMasterVolumeLevelScalar() * 100
                vol.SetMasterVolumeLevelScalar(max(0.0, (atual - 10) / 100), None)
            elif acao == "definir":
                vol.SetMasterVolumeLevelScalar(max(0, min(100, nivel)) / 100, None)
            elif acao == "mudo":
                vol.SetMute(not vol.GetMute(), None)
            return f"Volume agora em {round(vol.GetMasterVolumeLevelScalar() * 100)}%."
        except ImportError:
            return "Erro: no Windows instale 'pycaw' e 'comtypes' (pip install pycaw comtypes)."

    # Linux: tenta pactl (PulseAudio/PipeWire), depois amixer (ALSA)
    pactl = _which("pactl")
    if pactl:
        if acao == "aumentar":
            _run([pactl, "set-sink-volume", "@DEFAULT_SINK@", "+10%"])
        elif acao == "diminuir":
            _run([pactl, "set-sink-volume", "@DEFAULT_SINK@", "-10%"])
        elif acao == "definir":
            _run([pactl, "set-sink-volume", "@DEFAULT_SINK@", f"{max(0, min(100, nivel))}%"])
        elif acao == "mudo":
            _run([pactl, "set-sink-mute", "@DEFAULT_SINK@", "toggle"])
        return "Volume: " + (_run([pactl, "get-sink-volume", "@DEFAULT_SINK@"]) or "ajustado")

    amixer = _which("amixer")
    if amixer:
        if acao == "aumentar":
            _run([amixer, "set", "Master", "10%+"])
        elif acao == "diminuir":
            _run([amixer, "set", "Master", "10%-"])
        elif acao == "definir":
            _run([amixer, "set", "Master", f"{max(0, min(100, nivel))}%"])
        elif acao == "mudo":
            _run([amixer, "set", "Master", "toggle"])
        return "Volume: " + _run([amixer, "get", "Master"])

    return "Erro: nenhum controlador de volume encontrado (pactl/amixer)."


def tirar_print(caminho: str = "") -> str:
    """Captura a tela e salva como imagem PNG.

    Args:
        caminho: Onde salvar a imagem (padrão: "print_AAAAMMDD_HHMMSS.png" na pasta atual).
    """
    if not caminho:
        caminho = datetime.now().strftime("print_%Y%m%d_%H%M%S.png")
    caminho = os.path.expanduser(caminho)

    if IS_WINDOWS:
        try:
            import mss
            with mss.mss() as sct:
                img = sct.grab(sct.monitors[1])
                mss.tools.to_png(img.rgb, img.size, output=caminho)
            return f"Print salvo em '{caminho}'."
        except ImportError:
            return "Erro: no Windows instale 'mss' (pip install mss)."

    import_ = _which("gnome-screenshot", "spectacle", "scrot", "grim", "maim")
    if import_:
        cmd = [import_]
        if import_ == "scrot":
            cmd.append(caminho)
        elif import_ == "maim":
            cmd.append(caminho)
        elif import_ == "grim":
            cmd.append(caminho)
        else:  # gnome-screenshot / spectacle
            cmd += ["-f", caminho]
        saida = _run(cmd, timeout=20)
        if os.path.isfile(caminho):
            return f"Print salvo em '{caminho}'."
        return f"Ferramenta executou mas não confirmou o arquivo: {saida}"

    # fallback: importa PIL ImageGrab (funciona em alguns desktops Linux via X11)
    try:
        from PIL import ImageGrab
        ImageGrab.grab().save(caminho)
        return f"Print salvo em '{caminho}'."
    except Exception as e:
        return f"Erro: nenhuma ferramenta de screenshot disponível ({e}). Instale scrot ou gnome-screenshot."


# ---------------- REDE ----------------
def velocidade_ping(host: str = "8.8.8.8") -> str:
    """Testa a conexão com a internet fazendo ping em um servidor.

    Args:
        host: Endereço para testar (padrão: 8.8.8.8, DNS do Google).
    """
    if IS_WINDOWS:
        saida = _run(["ping", "-n", "4", host], timeout=20)
    else:
        saida = _run(["ping", "-c", "4", host], timeout=20)
    return saida


# ---------------- ÁREA DE TRANSFERÊNCIA ----------------
def copiar_texto(texto: str) -> str:
    """Copia um texto para a área de transferência do usuário.

    Args:
        texto: O texto completo a ser copiado.
    """
    try:
        import pyperclip
        pyperclip.copy(texto)
        return f"Texto copiado ({len(texto)} caracteres)."
    except Exception as e:
        return f"Erro ao copiar para a área de transferência: {e}"


# ---------------- AUTO-TESTE ----------------
if __name__ == "__main__":
    print(f"Sistema detectado: {OS_NAME} (Windows={IS_WINDOWS}, Linux={IS_LINUX}, Mac={IS_MAC})")
    print()
    print(sistema_info())
    print()
    print(data_hora())
