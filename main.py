import collections
import inspect
import json
import os
import queue
import random
import re
import shutil
import subprocess
import threading
import time
import types
import typing

import numpy as np

# Carrega variáveis de ambiente caso exista um arquivo .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI


def _setup_portaudio():
    import ctypes.util

    _orig_find_library = ctypes.util.find_library
    if _orig_find_library("portaudio"):
        return

    here = os.path.dirname(os.path.abspath(__file__))
    candidatos = (
        os.path.join(here, ".portaudio-libs"),
        os.path.expanduser("~/.local/lib"),
        "/usr/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/run/host/usr/lib",
    )
    for pasta in candidatos:
        for nome in ("libportaudio.so.2", "libportaudio.so"):
            caminho = os.path.join(pasta, nome)
            if os.path.isfile(caminho):
                os.environ["LD_LIBRARY_PATH"] = (
                    pasta + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                )

                def _find(name, _orig=_orig_find_library, _p=caminho):
                    return _p if name == "portaudio" else _orig(name)

                ctypes.util.find_library = _find
                return
    raise OSError(
        "PortAudio não encontrada. Instale o pacote portaudio "
        "(ex.: sudo pacman -S portaudio)."
    )


_setup_portaudio()
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
from vosk import KaldiRecognizer, Model as VoskModel, SetLogLevel

# Silencia logs internos do Vosk
SetLogLevel(-1)

try:
    import tools as _tools
except ImportError:
    _tools = None


# ---------------- CARREGAMENTO DA PERSONALIDADE ----------------
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_PERSONALIDADE = os.path.join(DIRETORIO_ATUAL, "personalidade.txt")

PROMPT_PADRAO = """Você é o Samuel V2, um robô assistente pessoal de inteligência artificial que veio para substituir o Samuel humano."""


def carregar_personalidade() -> str:
    if os.path.isfile(ARQUIVO_PERSONALIDADE):
        try:
            with open(ARQUIVO_PERSONALIDADE, "r", encoding="utf-8") as f:
                conteudo = f.read().strip()
                if conteudo:
                    print(f"📖 Personalidade carregada de '{ARQUIVO_PERSONALIDADE}' ({len(conteudo)} caracteres).")
                    return conteudo
        except Exception as e:
            print(f"Aviso: Falha ao ler '{ARQUIVO_PERSONALIDADE}': {e}. Usando prompt padrão.")

    # Se o arquivo não existir, cria com o modelo padrão para conveniência
    try:
        with open(ARQUIVO_PERSONALIDADE, "w", encoding="utf-8") as f:
            f.write(PROMPT_PADRAO.strip())
        print(f"📄 Arquivo '{ARQUIVO_PERSONALIDADE}' não existia e foi gerado com o prompt padrão.")
    except Exception:
        pass

    return PROMPT_PADRAO


SYSTEM_PROMPT = carregar_personalidade()


# ---------------- CLASSE DE BENCHMARK DE TEMPO ----------------
class Benchmark:
    def __init__(self, t_fim_fala: float):
        self.t_fim_fala = t_fim_fala          # Fim da fala do usuário (detectado pelo VAD)
        self.t_transcricao = 0.0              # Fim do Whisper
        self.t_llm_envio = 0.0                # Momento em que a requisição ao LLM foi disparada
        self.t_llm_ttft = 0.0                 # Time To First Token (1º token recebido do LLM)
        self.t_tts_primeira_frase = 0.0       # 1ª frase completa montada e enviada ao Piper
        self.t_tts_primeiro_audio = 0.0       # 1º bloco de áudio gerado pelo Piper
        self.t_inicio_audio = 0.0             # 1º bloco enviado para a placa de som

    def registrar_inicio_audio(self):
        if self.t_inicio_audio == 0.0:
            self.t_inicio_audio = time.time()

    def exibir_relatorio(self):
        if self.t_inicio_audio == 0.0:
            return

        total = self.t_inicio_audio - self.t_fim_fala
        t_transcr = (self.t_transcricao - self.t_fim_fala) if self.t_transcricao else 0.0
        ttft = (self.t_llm_ttft - self.t_llm_envio) if (self.t_llm_ttft and self.t_llm_envio) else 0.0
        t_frase = (self.t_tts_primeira_frase - self.t_llm_ttft) if (self.t_tts_primeira_frase and self.t_llm_ttft) else 0.0
        t_piper = (self.t_tts_primeiro_audio - self.t_tts_primeira_frase) if (self.t_tts_primeiro_audio and self.t_tts_primeira_frase) else 0.0
        t_buffer = (self.t_inicio_audio - self.t_tts_primeiro_audio) if (self.t_inicio_audio and self.t_tts_primeiro_audio) else 0.0

        print("\n" + "=" * 55)
        print("⏱️  BENCHMARK DE LATÊNCIA (Fim da Fala → Início da Voz)")
        print("=" * 55)
        print(f"├─ 🎙️ Transcrição (Whisper)       : {t_transcr * 1000:>6.1f} ms ({t_transcr:.2f}s)")
        print(f"├─ 🌐 1º Token LLM (OpenRouter)   : {ttft * 1000:>6.1f} ms ({ttft:.2f}s)")
        print(f"├─ 📝 Formar 1ª Frase             : {t_frase * 1000:>6.1f} ms ({t_frase:.2f}s)")
        print(f"├─ 🗣️ Síntese Piper (1º Bloco)    : {t_piper * 1000:>6.1f} ms ({t_piper:.2f}s)")
        print(f"├─ 🔊 Buffer de Saída Áudio       : {t_buffer * 1000:>6.1f} ms ({t_buffer:.2f}s)")
        print("—" * 55)
        print(f"🚀 TEMPO TOTAL ATÉ A FALA INICIAR : {total:>6.2f}s ({total * 1000:.0f} ms)")
        print("=" * 55 + "\n")


# ---------------- CONFIGURAÇÕES ----------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "ERRO DE AUTENTICAÇÃO: A variável OPENROUTER_API_KEY não foi encontrada!\n"
        "Defina-a no seu terminal com 'export OPENROUTER_API_KEY=sk-or-v1-...' "
        "ou crie um arquivo .env no diretório do projeto."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Samuel Assistant",
    },
)

MODELO_LLM = "google/gemma-4-26b-a4b-it"
MAX_TOKENS = 300
MAX_RODADAS_TOOLS = 4

WHISPER_MODELO = "base"  # tente "small" para transcrever melhor em português
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"
PIPER_VOZ = "pt_BR-faber-medium.onnx"

VOSK_MODEL_PATH = "vosk-model-small-pt-0.3"

SAMPLE_RATE = 16000
BLOCO = 512
DUR_BLOCO = BLOCO / SAMPLE_RATE
SILENCIO_FIM = 0.4
ESPERA_FALA_MAX = 8.0
GRAVACAO_MAX = 30.0
LIMIAR_MIN = 0.012
RABO_FALA = 0.2

# Ativação: espera a palavra "Samuel" terminar antes de responder
ATIVACAO_SILENCIO_FIM = 0.2   # silêncio que indica que a palavra acabou
ATIVACAO_ESPERA_MAX = 1.5     # tempo máximo esperando a palavra acabar
ATIVACAO_PAUSA_POS = 0.15     # folga depois da frase de ativação (evita eco)

VAD_MODELO = "silero_vad.onnx"
VAD_LIMIAR = 0.5

audio_lock = threading.Lock()
historico_lock = threading.Lock()

ultimo_clipboard = ""


# ---------- VAD NEURAL (SILERO ONNX, SEM TORCH) ----------
class VADNeural:
    def __init__(self, caminho: str, taxa: int = SAMPLE_RATE):
        import onnxruntime as ort

        opcoes = ort.SessionOptions()
        opcoes.intra_op_num_threads = 1
        opcoes.inter_op_num_threads = 1
        opcoes.log_severity_level = 3
        self.sessao = ort.InferenceSession(
            caminho, sess_options=opcoes, providers=["CPUExecutionProvider"]
        )
        self.saidas = [s.name for s in self.sessao.get_outputs()]
        self.v5 = any(i.name == "state" for i in self.sessao.get_inputs())
        self.taxa = np.array(taxa, dtype=np.int64)
        self.reset()

    def reset(self):
        self.estado = np.zeros((2, 1, 128), dtype=np.float32)
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)
        self.contexto = np.zeros(64, dtype=np.float32)

    def probabilidade(self, bloco: np.ndarray) -> float:
        entrada = np.concatenate((self.contexto, bloco)).astype(np.float32)[None, :]
        self.contexto = entrada[0, -64:].copy()
        if self.v5:
            saida, self.estado = self.sessao.run(
                self.saidas, {"input": entrada, "state": self.estado, "sr": self.taxa}
            )
        else:
            saida, self.h, self.c = self.sessao.run(
                self.saidas, {"input": entrada, "sr": self.taxa, "h": self.h, "c": self.c}
            )
        return float(saida[0][0])


# ---------- LEITURA DO CLIPBOARD ----------
def obter_clipboard() -> str:
    try:
        import pyperclip
        txt = pyperclip.paste()
        if txt and txt.strip():
            return txt.strip()
    except Exception:
        pass

    if shutil.which("xclip"):
        try:
            res = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=0.8,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass

    return ""


# ---------- SISTEMA DE TIMERS / LEMBRETES ----------
def disparar_lembrete(motivo: str):
    try:
        print(f"\n\n⏰ [TIMER DISPARADO]: {motivo}")

        prompt_alerta = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"[ALERTA DE TIMER: O tempo que o usuário pediu acabou de esgotar! Motivo: '{motivo}'. "
                    "Gere uma fala de uma ou duas frases avisando ele que o tempo acabou no seu estilo tsundere/reclamão habitual.]"
                ),
            },
        ]

        resposta = client.chat.completions.create(
            model=MODELO_LLM,
            messages=prompt_alerta,
            max_tokens=120,
            extra_body={"reasoning": {"enabled": False}},
            timeout=20,
        )

        texto_fala = resposta.choices[0].message.content or ""
        texto_fala = limpar(texto_fala)
        if not texto_fala:
            texto_fala = f"Ei! O tempo de {motivo} acabou. Vai lá resolver antes que dê problema."

        print(f"\nIA (Alarme): {texto_fala}\n")

        with historico_lock:
            historico.append({"role": "assistant", "content": f"[Aviso de Timer]: {texto_fala}"})

        with audio_lock:
            for chunk in voz.synthesize(texto_fala):
                saida.write(chunk.audio_int16_bytes)

    except Exception as e:
        print(f"\n[Erro ao disparar timer]: {e}")


def criar_lembrete(minutos: float, motivo: str) -> str:
    """Agenda um lembrete/timer que avisa o usuário por voz quando o tempo acabar.

    Args:
        minutos: Quantos minutos até o lembrete disparar.
        motivo: Motivo do lembrete (ex.: "tirar o bolo do forno").
    """
    minutos_num = float(minutos)
    segundos = max(1.0, minutos_num * 60.0)
    timer = threading.Timer(segundos, disparar_lembrete, args=(motivo,))
    timer.daemon = True
    timer.start()
    return f"Lembrete agendado com sucesso para daqui a {minutos_num} minutos. Motivo: {motivo}."


# ---------- FERRAMENTAS ----------
TIPOS_JSON = {str: "string", int: "integer", float: "number", bool: "boolean",
              list: "array", dict: "object"}


def _tipo_json(anotacao) -> str:
    origem = typing.get_origin(anotacao)
    if origem in (typing.Union, getattr(types, "UnionType", None)):
        args = [a for a in typing.get_args(anotacao) if a is not type(None)]
        anotacao = args[0] if args else str
        origem = typing.get_origin(anotacao)
    return TIPOS_JSON.get(origem or anotacao, "string")


def _ler_docstring(fn):
    doc = inspect.getdoc(fn) or ""
    partes = re.split(r"\n\s*(?:Args|Parâmetros):\s*\n", doc, maxsplit=1)
    descricao = partes[0].strip()
    args = {}
    if len(partes) > 1:
        for linha in partes[1].splitlines():
            m = re.match(r"\s*(\w+)\s*(?:\([^)]*\))?:\s*(.+)", linha)
            if m:
                args[m.group(1)] = m.group(2).strip()
    return descricao, args


def _extrair_schema(fn):
    descricao, docs_args = _ler_docstring(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    props, obrigatorios = {}, []
    for p in inspect.signature(fn).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        prop = {"type": _tipo_json(hints.get(p.name, str))}
        if p.name in docs_args:
            prop["description"] = docs_args[p.name]
        props[p.name] = prop
        if p.default is inspect.Parameter.empty:
            obrigatorios.append(p.name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": descricao or fn.__name__,
            "parameters": {"type": "object", "properties": props, "required": obrigatorios},
        },
    }


def carregar_ferramentas():
    funcoes, schemas = {}, []
    ferramentas_base = [criar_lembrete]
    for fn in ferramentas_base:
        schemas.append(_extrair_schema(fn))
        funcoes[fn.__name__] = fn

    if _tools is not None:
        for nome, fn in inspect.getmembers(_tools, inspect.isfunction):
            if nome.startswith("_") or fn.__module__ != _tools.__name__:
                continue
            schemas.append(_extrair_schema(fn))
            funcoes[nome] = fn

    return funcoes, schemas


def executar_ferramenta(nome: str, argumentos_json: str) -> str:
    fn = FUNCOES.get(nome)
    if fn is None:
        return f"Erro: a ferramenta '{nome}' não existe."
    try:
        args = json.loads(argumentos_json) if argumentos_json else {}
        resultado = fn(**args)
    except Exception as e:
        return f"Erro ao executar '{nome}': {e}"
    if not isinstance(resultado, str):
        resultado = json.dumps(resultado, ensure_ascii=False, default=str)
    return resultado


FUNCOES, FERRAMENTAS = carregar_ferramentas()
if FERRAMENTAS:
    print("Ferramentas carregadas:", ", ".join(FUNCOES))
else:
    print("Nenhuma ferramenta carregada.")


# ---------- CARREGAMENTO DOS MODELOS ----------
print("Carregando Vosk (Wakeword)...")
if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError(
        f"Modelo do Vosk não encontrado em '{VOSK_MODEL_PATH}'. "
        "Baixe e descompacte de https://alphacephei.com/vosk/models"
    )
vosk_model = VoskModel(VOSK_MODEL_PATH)

print("Carregando Whisper...")
whisper = WhisperModel(
    WHISPER_MODELO, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE,
    cpu_threads=min(4, os.cpu_count() or 4),
)
list(whisper.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"), language="pt")[0])

print("Carregando Piper...")
voz = PiperVoice.load(PIPER_VOZ)
for _ in voz.synthesize("Pronto."):
    pass
saida = sd.RawOutputStream(samplerate=voz.config.sample_rate, channels=1, dtype="int16", latency="low")
saida.start()

print("Carregando VAD neural (Silero ONNX)...")
VAD_NEURAL = None
try:
    VAD_NEURAL = VADNeural(VAD_MODELO)
    VAD_NEURAL.probabilidade(np.zeros(BLOCO, dtype=np.float32))
    VAD_NEURAL.reset()
    print("VAD neural pronto.")
except Exception as e:
    print(f"AVISO: VAD neural indisponível ({e}).")
    print("       Rode '.venv/bin/python setup.py --vad' para instalar. Usando VAD por energia.")

historico = [{"role": "system", "content": SYSTEM_PROMPT}]


# ---------- FRASES DE ATIVAÇÃO (PRÉ-SINTETIZADAS) ----------
FRASES_ATIVACAO = [
    "O que quer agora?",
    "Fala logo.",
    "Que foi?",
    "Hm? Fala.",
    "Estou ouvindo. Infelizmente.",
    "Diz aí.",
]
print("Pré-sintetizando frases de ativação...")
AUDIOS_ATIVACAO = [
    b"".join(c.audio_int16_bytes for c in voz.synthesize(f)) for f in FRASES_ATIVACAO
]
_ultima_ativacao = -1


def falar_ativacao():
    """Toca uma frase de ativação aleatória (sem repetir a anterior)."""
    global _ultima_ativacao
    opcoes = [i for i in range(len(AUDIOS_ATIVACAO)) if i != _ultima_ativacao]
    i = random.choice(opcoes)
    _ultima_ativacao = i
    print(f"IA: {FRASES_ATIVACAO[i]}")
    with audio_lock:
        saida.write(AUDIOS_ATIVACAO[i])
    time.sleep(ATIVACAO_PAUSA_POS)  # deixa o final do áudio terminar de tocar


def _esvaziar(fila):
    """Descarta o que o microfone captou enquanto o Samuel falava."""
    while True:
        try:
            fila.get_nowait()
        except queue.Empty:
            return


def _esperar_fim_da_palavra(fila, detectar):
    """Espera a palavra 'Samuel' terminar (um curto silêncio do VAD)."""
    silencio, total = 0.0, 0.0
    while total < ATIVACAO_ESPERA_MAX:
        bloco = fila.get()
        total += DUR_BLOCO
        if detectar(bloco):
            silencio = 0.0
        else:
            silencio += DUR_BLOCO
            if silencio >= ATIVACAO_SILENCIO_FIM:
                return


# ---------- ESCUTA: VAD NEURAL -> VOSK -> (FRASE) -> GRAVA -> WHISPER ----------
HANGOVER_VOSK = 0.5


class _DetectorFala:
    def __init__(self):
        self.vad = VAD_NEURAL
        self.ruido = []
        if self.vad is not None:
            self.vad.reset()

    def reiniciar(self):
        """Zera o estado do VAD neural (mantém a calibração de ruído do VAD por energia)."""
        if self.vad is not None:
            self.vad.reset()

    def __call__(self, bloco: np.ndarray) -> bool:
        if self.vad is not None:
            return self.vad.probabilidade(bloco) >= VAD_LIMIAR
        nivel = float(np.sqrt(np.mean(bloco ** 2)))
        if len(self.ruido) < 10:
            self.ruido.append(nivel)
            return False
        limiar = min(max(LIMIAR_MIN, float(np.mean(self.ruido)) * 3), 0.08)
        return nivel > limiar


def _bytes_vosk(bloco: np.ndarray) -> bytes:
    return (bloco * 32767.0).astype(np.int16).tobytes()


def escutar_comando():
    fila = queue.Queue()

    def callback(indata, frames, tempo, status):
        fila.put(indata[:, 0].copy())

    detectar = _DetectorFala()
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE, json.dumps(["samuel", "[unk]"]))
    pendente = collections.deque(maxlen=int(HANGOVER_VOSK / DUR_BLOCO))
    ativo = VAD_NEURAL is None
    silencio_vad = 0.0

    def alimentar_vosk() -> bool:
        while pendente:
            bloco = pendente.popleft()
            if rec.AcceptWaveform(_bytes_vosk(bloco)):
                if "samuel" in json.loads(rec.Result()).get("text", "").lower():
                    return True
            elif "samuel" in json.loads(rec.PartialResult()).get("partial", "").lower():
                return True
        return False

    print("\n🟢 [VAD] Escutando... (Fale 'Samuel' para ativar)")
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=BLOCO, latency="low", callback=callback,
    ):
        while True:
            bloco = fila.get()
            fala = detectar(bloco)
            pendente.append(bloco)

            if fala:
                silencio_vad = 0.0
                ativo = True
            elif ativo and VAD_NEURAL is not None:
                silencio_vad += DUR_BLOCO
                if silencio_vad >= HANGOVER_VOSK:
                    ativo = False
                    pendente.clear()

            if ativo and alimentar_vosk():
                print("\n⚡ Wakeword 'Samuel' detectada!")
                # 1) espera você terminar de falar "Samuel"
                _esperar_fim_da_palavra(fila, detectar)
                # 2) responde na hora com uma frase pronta (sem LLM e sem Whisper)
                falar_ativacao()
                # 3) joga fora o que o microfone captou durante a frase
                _esvaziar(fila)
                detectar.reiniciar()
                # 4) só agora começa a gravar o comando
                print("🎙️ Ouvindo seu comando...")
                return _gravar_comando(fila, detectar)


def _recorte(pedacos, ultima_fala):
    fim = min(len(pedacos), ultima_fala + int(RABO_FALA / DUR_BLOCO) + 1)
    return np.concatenate(pedacos[:fim]) if fim else np.zeros(0, dtype="float32")


def _transcrever_em_thread(audio):
    caixa = {}

    def rodar():
        try:
            caixa["texto"] = transcrever(audio)
        except Exception:
            caixa["texto"] = ""

    caixa["thread"] = threading.Thread(target=rodar, daemon=True)
    caixa["thread"].start()
    return caixa


def _gravar_comando(fila, detectar):
    pedacos = []
    falou = False
    ultima_fala = 0
    silencio = 0.0
    inicio = time.time()
    especulativa = None

    def especular():
        nonlocal especulativa
        if especulativa is not None:
            if especulativa["marca"] == ultima_fala or especulativa["caixa"]["thread"].is_alive():
                return
        especulativa = {
            "marca": ultima_fala,
            "caixa": _transcrever_em_thread(_recorte(pedacos, ultima_fala)),
        }

    while True:
        bloco = fila.get()
        pedacos.append(bloco)
        decorrido = time.time() - inicio

        if detectar(bloco):
            falou, silencio, ultima_fala = True, 0.0, len(pedacos)
        elif falou:
            silencio += DUR_BLOCO
            if silencio >= SILENCIO_FIM:
                break
            if silencio >= 2 * DUR_BLOCO:
                especular()
        elif decorrido > ESPERA_FALA_MAX:
            return np.zeros(0, dtype="float32"), "", time.time()

        if decorrido > GRAVACAO_MAX:
            break

    fim = time.time() - max(silencio, 0.0)
    audio = _recorte(pedacos, ultima_fala)

    if especulativa is not None and especulativa["marca"] == ultima_fala:
        caixa = especulativa["caixa"]
        caixa["thread"].join()
        if caixa.get("texto"):
            return audio, caixa["texto"], fim
    return audio, "", fim


# ---------- TRANSCRIÇÃO (WHISPER) ----------
def transcrever(audio: np.ndarray) -> str:
    segmentos, _ = whisper.transcribe(
        audio, language="pt", beam_size=1,
        condition_on_previous_text=False, without_timestamps=True,
        initial_prompt="Conversa em português do Brasil.",
    )
    texto = " ".join(s.text.strip() for s in segmentos).strip()

    # Remove um "Samuel" que tenha sobrado no começo
    texto = re.sub(
        r"^\W*(?:ei\s+|olá\s+|oxa\s+|oh\s+)*samuel[,\s.\-!]*\s*",
        "",
        texto,
        flags=re.IGNORECASE,
    ).strip()

    return texto


# ---------- STREAM LLM (VIA OPENROUTER CLIENT) ----------
def stream_llm(mensagens, chamadas: dict, bench: Benchmark, usar_tools: bool = True):
    kwargs = {
        "model": MODELO_LLM,
        "messages": mensagens,
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "extra_body": {
            "reasoning": {"enabled": False},
            "provider": {"sort": "latency"},
        },
        "timeout": 60,
    }
    if usar_tools and FERRAMENTAS:
        kwargs["tools"] = FERRAMENTAS
        kwargs["tool_choice"] = "auto"

    bench.t_llm_envio = time.time()
    resposta_stream = client.chat.completions.create(**kwargs)

    for chunk in resposta_stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # Registra o primeiro token recebido (TTFT)
        if (delta.content or delta.tool_calls) and bench.t_llm_ttft == 0.0:
            bench.t_llm_ttft = time.time()

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                slot = chamadas.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

        if delta.content:
            yield delta.content


CORTE_NORMAL = re.compile(r"(?<=[.!?…])\s+|\n+")
CORTE_RAPIDO = re.compile(r"(?<=[.!?…,;:])\s+|\n+")


def frases(tokens, bench: Benchmark):
    buf, primeira = "", True
    for t in tokens:
        buf += t
        while True:
            padrao = CORTE_RAPIDO if primeira else CORTE_NORMAL
            minimo = 20 if primeira else 1
            corte = next((m for m in padrao.finditer(buf) if m.start() >= minimo), None)
            if not corte:
                break
            if primeira and bench.t_tts_primeira_frase == 0.0:
                bench.t_tts_primeira_frase = time.time()
            yield buf[:corte.start()]
            buf = buf[corte.end():]
            primeira = False
    if buf.strip():
        if bench.t_tts_primeira_frase == 0.0:
            bench.t_tts_primeira_frase = time.time()
        yield buf


def limpar(texto: str) -> str:
    texto = re.sub(r"[*_#`>~]+", " ", texto).strip()
    return texto if re.search(r"\w", texto) else ""


def trabalhador_tts(fila_texto, fila_audio, bench: Benchmark):
    while True:
        frase = fila_texto.get()
        if frase is None:
            fila_audio.put(None)
            return
        for chunk in voz.synthesize(frase):
            if bench.t_tts_primeiro_audio == 0.0:
                bench.t_tts_primeiro_audio = time.time()
            fila_audio.put(chunk.audio_int16_bytes)


def trabalhador_play(fila_audio, bench: Benchmark):
    while True:
        dados = fila_audio.get()
        if dados is None:
            return
        bench.registrar_inicio_audio()
        with audio_lock:
            saida.write(dados)


def uma_rodada(fila_texto, usar_tools: bool, bench: Benchmark):
    partes, chamadas = [], {}

    def tokens_visiveis():
        with historico_lock:
            mensagens_envio = list(historico)
        for t in stream_llm(mensagens_envio, chamadas, bench, usar_tools):
            partes.append(t)
            print(t, end="", flush=True)
            yield t

    print("IA: ", end="", flush=True)
    for frase in frases(tokens_visiveis(), bench):
        f = limpar(frase)
        if f:
            fila_texto.put(f)
    print()
    return "".join(partes).strip(), chamadas


# ---------- PROCESSAMENTO DA RESPOSTA ----------
def responder(texto_usuario: str, bench: Benchmark):
    global ultimo_clipboard

    clip_atual = obter_clipboard()
    if clip_atual and clip_atual != ultimo_clipboard:
        ultimo_clipboard = clip_atual
        clip_cortado = clip_atual[:2000] + ("\n[...truncado]" if len(clip_atual) > 2000 else "")
        conteudo_envio = f"[Novo conteúdo copiado no Ctrl+C:\n{clip_cortado}\n]\nUsuário: {texto_usuario}"
        print(f"\n[Clipboard detectado e anexado: {len(clip_atual)} chars]")
    else:
        conteudo_envio = texto_usuario

    with historico_lock:
        tamanho_antes = len(historico)
        historico.append({"role": "user", "content": conteudo_envio})

    fila_texto, fila_audio = queue.Queue(), queue.Queue()
    th_tts = threading.Thread(target=trabalhador_tts, args=(fila_texto, fila_audio, bench), daemon=True)
    th_play = threading.Thread(target=trabalhador_play, args=(fila_audio, bench), daemon=True)
    th_tts.start()
    th_play.start()

    try:
        for rodada in range(MAX_RODADAS_TOOLS + 1):
            texto, chamadas = uma_rodada(fila_texto, usar_tools=rodada < MAX_RODADAS_TOOLS, bench=bench)
            if not chamadas:
                if texto:
                    with historico_lock:
                        historico.append({"role": "assistant", "content": texto})
                break

            ordenadas = [c for _, c in sorted(chamadas.items())]
            for i, c in enumerate(ordenadas):
                c["id"] = c["id"] or f"call_{rodada}_{i}"

            with historico_lock:
                historico.append({
                    "role": "assistant",
                    "content": texto or None,
                    "tool_calls": [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                        for c in ordenadas
                    ],
                })

            for c in ordenadas:
                print(f"[ferramenta] {c['name']}({c['arguments']})")
                resultado = executar_ferramenta(c["name"], c["arguments"])
                print(f"[resultado] {resultado}")
                with historico_lock:
                    historico.append({"role": "tool", "tool_call_id": c["id"], "content": resultado})
    except Exception as e:
        print(f"\nErro no LLM: {e}")
        with historico_lock:
            del historico[tamanho_antes:]
    finally:
        fila_texto.put(None)
        th_tts.join()
        th_play.join()

    # Exibe o painel de tempos após o áudio ter sido reproduzido
    bench.exibir_relatorio()


# ---------- LOOP PRINCIPAL ----------
def main():
    print("\nSistema inicializado com sucesso!")
    while True:
        try:
            audio, texto, fim_fala = escutar_comando()
            if len(audio) < SAMPLE_RATE * 0.2:
                print("Fala muito curta ou silêncio detectado.")
                continue

            bench = Benchmark(t_fim_fala=fim_fala)

            if not texto:
                texto = transcrever(audio)

            bench.t_transcricao = time.time()

            if not texto:
                print("Não entendi nada do que disse.")
                continue

            print(f"Você: {texto}")
            responder(texto, bench)

        except KeyboardInterrupt:
            print("\nEncerrando Samuel...")
            break
        except Exception as e:
            print(f"Erro no ciclo: {e}")


if __name__ == "__main__":
    main()