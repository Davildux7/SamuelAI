# 🤖 Samuel V2

Assistente de voz pessoal em Python que **substitui o Samuel humano** (segundo ele mesmo). Fale **"Samuel"**, ele responde na hora, ouve seu comando e responde falando — com ferramentas reais de sistema operacional nas mãos.

> Pipeline 100% local para voz (wake word → transcrição → TTS), LLM via OpenRouter.

## Como funciona

```
 🎙️ Microfone
   │  VAD neural (Silero ONNX) detecta que você falou
   ▼
 Vosk (wakeword)  ──►  "Samuel" detectado
   │                    └─► responde com uma frase de ativação pré-sintetizada (sem LLM!)
   ▼
 Gravação com fim de fala automático (silêncio 0,4 s)
   ▼
 Faster-Whisper (transcrição PT, já começa durante a gravação — especulativo)
   ▼
 LLM via OpenRouter (streaming, com tool calling)
   ▼
 Piper TTS (sintetiza frase por frase enquanto o LLM ainda fala)
   ▼
 🔊 Placa de som
```

Detalhes de latência:
- **Wake word** não usa Whisper — Vosk é leve e roda em tempo real.
- **Frases de ativação** ("O que quer agora?", "Fala logo."…) são pré-sintetizadas no boot, zero espera.
- **Transcrição especulativa**: o Whisper já transcreve enquanto você ainda fala.
- **TTS em streaming**: cada frase completa vai pro Piper antes do LLM terminar.
- No fim de cada resposta aparece um **benchmark** com o tempo de cada etapa.

## Ferramentas (tool calling)

O LLM pode chamar as ferramentas do `tools.py` (até 4 rodadas por resposta):

| Ferramenta | O que faz |
|---|---|
| `sistema_info` | CPU, RAM, bateria, versão do SO |
| `data_hora` | Data e hora em português |
| `abrir_site` | Abre URL no navegador |
| `abrir_arquivo_ou_app` | Abre apps, arquivos ou pastas |
| `listar_arquivos` / `ler_arquivo` / `escrever_arquivo` | Operações de arquivo |
| `processos_ativos` / `encerrar_processo` | Lista e mata processos |
| `controlar_volume` | Aumenta/diminui/muda/defini volume |
| `tirar_print` | Captura de tela |
| `velocidade_ping` | Testa conexão |
| `copiar_texto` | Copia texto pra área de transferência |
| `criar_lembrete` | Timer que avisa por voz ("me lembre em 10 min de...") |

Tudo com **detecção de OS**: no Windows usa `taskkill`/pycaw/`startfile`, no Linux usa `pkill`/`pactl`/`xdg-open`. Para adicionar ferramentas novas, basta criar uma função pública em `tools.py` com type hints e docstring `Args:` — o `main.py` registra automaticamente no LLM.

Extras: conteúdo novo copiado no **Ctrl+C** é automaticamente anexado à próxima mensagem.

## Instalação

### 1. Requisitos

- Python 3.10+
- **Linux**: `portaudio` e `ffmpeg` (o setup instala via pacman/apt/dnf/zypper)
- **Windows**: nada além do Python — o setup orienta o `winget install Gyan.FFmpeg` (opcional)

### 2. Setup automático

```bash
cd SamuelV2
python -m venv .venv && source .venv/bin/activate   # Linux (recomendado)
# .venv\Scripts\activate                            # Windows

python setup.py
```

O `setup.py` faz tudo: pacotes de sistema, `pip install -r requirements.txt` e download dos modelos (~500 MB no total). Opções:

```bash
python setup.py --modelos      # só baixa os modelos de voz
python setup.py --vad          # só instala o silero_vad.onnx
python setup.py --sem-sistema  # pula os pacotes do sistema (sem sudo)
```

### 3. Chave de API

Crie um arquivo `.env` na pasta do projeto (ou exporte no terminal):

```
OPENROUTER_API_KEY=sk-or-v1-...
```

### 4. Rodar

```bash
python main.py
```

Fale **"Samuel"** e espere ele responder. 🟢 = escutando.

## Modelos usados (baixados pelo setup)

| Arquivo | Uso | Origem |
|---|---|---|
| `vosk-model-small-pt-0.3/` | Wake word "Samuel" | [alphacephei.com/vosk](https://alphacephei.com/vosk/models) |
| `silero_vad.onnx` | Detecção de fala (VAD neural) | [silero-vad](https://github.com/snakers4/silero-vad) |
| `pt_BR-faber-medium.onnx` | Voz do Piper TTS (pt-BR) | [piper-voices](https://huggingface.co/rhasspy/piper-voices) |

## Configurações (`main.py`)

```python
MODELO_LLM = "google/gemma-4-26b-a4b-it"   # modelo no OpenRouter
WHISPER_MODELO = "base"                    # troque p/ "small" p/ melhor PT
PIPER_VOZ = "pt_BR-faber-medium.onnx"
SILENCIO_FIM = 0.4                         # silêncio que encerra o comando
```

## Personalidade

Edite `personalidade.txt` — o conteúdo vira o *system prompt* do assistente (criado com um padrão tsundere/reclamão se não existir).

## Solução de problemas

| Problema | Solução |
|---|---|
| `PortAudio não encontrada` | `sudo pacman -S portaudio` (Arch) / `sudo apt install portaudio19-dev` |
| `Modelo do Vosk não encontrado` | `python setup.py --modelos` |
| `AVISO: VAD neural indisponível` | `python setup.py --vad` (funciona mesmo sem, via energia) |
| Whisper erra muito em PT | Troque `WHISPER_MODELO` para `"small"` (mais lento) |
| Windows: volume/screenshot falham | `pip install pycaw comtypes mss` |
| Sem áudio na saída | Verifique o dispositivo padrão; o Piper usa a taxa do próprio modelo |
| Benchmark não aparece | Só aparece quando a resposta de fato falou algo |

## Estrutura

```
SamuelV2/
├── main.py            # pipeline completo de voz + LLM
├── tools.py           # ferramentas do LLM (multi-OS)
├── setup.py           # instalador (deps + modelos)
├── requirements.txt
├── personalidade.txt  # system prompt (criado no 1º run)
├── vosk-model-small-pt-0.3/
├── silero_vad.onnx
└── pt_BR-faber-medium.onnx
```
