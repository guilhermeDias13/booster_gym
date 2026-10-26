"""Extrai frames de um vídeo e grava numa pasta nomeada aqui.

O nome da pasta e a lista de vídeos ficam neste arquivo. Ao rodar, o programa
percorre os vídeos de exemplo em eval/videos_e1_download e
eval/videos_e3_download, salva os frames e monta um painel com todos.
"""

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuração — edite aqui
# ---------------------------------------------------------------------------

# Quantos frames gravar. PASSO = 1 deixa os frames seguidos;
# PASSO = 3 pega um frame, pula dois, e assim por diante.
N_FRAMES = 6
PASSO = 3

# Início da sequência. Se SEGUNDO_INICIAL estiver preenchido, ele vale;
# senão usa FRAME_INICIAL.
SEGUNDO_INICIAL = 40
FRAME_INICIAL = 50

# Zoom no centro do frame, só no painel. 1.0 mostra o quadro inteiro;
# 0.35 guarda o miolo e amplia cerca de 3 vezes.
ZOOM_CENTRO = 0.35

# Quantas linhas o painel mostra (uma linha = um vídeo). None = todas.
N_LINHAS = 1

# Legenda de cada linha: "abaixo", "acima" ou None para não escrever nada.
LEGENDA = None

# Raiz do resultado. Cada vídeo ganha uma subpasta com o nome definido abaixo.
PASTA_RESULTADO = "frames_extraidos"

# Pastas (relativas a eval/) onde estão os vídeos de exemplo.
PASTAS_DE_EXEMPLO = [
    "videos_e1_download",
    "videos_e3_download",
]

# Para processar um vídeo só, preencha os dois. None usa todos os exemplos.
VIDEO_ESPECIFICO = "videos_e1_download/E1_free_loss/env864.mp4"
NOME_PASTA_ESPECIFICA = "E1_free_loss_env864_s40"


def nome_da_pasta(video: Path) -> str:
    """Nome da pasta de saída, definido no programa: condicao_envNNN."""
    return f"{video.parent.name}_{video.stem}"


# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).resolve().parent


def listar_videos():
    """Devolve (caminho do vídeo, nome da pasta) para cada exemplo."""
    if VIDEO_ESPECIFICO:
        video = EVAL_DIR / VIDEO_ESPECIFICO
        return [(video, NOME_PASTA_ESPECIFICA)]

    pares = []
    for pasta in PASTAS_DE_EXEMPLO:
        for video in sorted((EVAL_DIR / pasta).rglob("*.mp4")):
            pares.append((video, nome_da_pasta(video)))
    return pares


def indices_dos_frames(fps: float) -> list[int]:
    """Índices pedidos: início no segundo configurado, de PASSO em PASSO."""
    if SEGUNDO_INICIAL is not None:
        inicio = round(SEGUNDO_INICIAL * fps)
    else:
        inicio = FRAME_INICIAL
    if PASSO < 1:
        raise SystemExit("PASSO precisa ser pelo menos 1")
    return [inicio + i * PASSO for i in range(N_FRAMES)]


def extrair_frames(video: Path, pasta: Path) -> list[Path]:
    """Grava os frames cujos índices estão em indices_dos_frames."""
    pasta.mkdir(parents=True, exist_ok=True)
    for antigo in pasta.glob("frame_*.png"):
        antigo.unlink()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"não abriu o vídeo: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"fps inválido em {video}")
    pedidos = indices_dos_frames(fps)
    querer = set(pedidos)
    ultimo = pedidos[-1]
    salvos = []
    indice = 0
    try:
        while indice <= ultimo:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"{video.name} acabou no frame {indice}; "
                    f"pedidos os frames {pedidos}"
                )
            if indice in querer:
                destino = pasta / f"frame_{indice:04d}.png"
                if not cv2.imwrite(str(destino), frame):
                    raise RuntimeError(f"falha ao gravar {destino}")
                salvos.append(destino)
                print(f"  frame {indice} = {indice / fps:.2f} s")
            indice += 1
    finally:
        cap.release()
    return salvos


def _fonte(tamanho: int) -> ImageFont.ImageFont:
    for caminho in (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ):
        if Path(caminho).is_file():
            return ImageFont.truetype(caminho, tamanho)
    return ImageFont.load_default()


def recortar_centro(img: Image.Image, fracao: float) -> Image.Image:
    """Mantém a fração central da imagem. 1.0 devolve o frame inteiro."""
    fracao = min(1.0, max(0.05, fracao))
    largura, altura = img.size
    recorte_l = max(1, round(largura * fracao))
    recorte_a = max(1, round(altura * fracao))
    esquerda = (largura - recorte_l) // 2
    topo = (altura - recorte_a) // 2
    return img.crop((esquerda, topo, esquerda + recorte_l, topo + recorte_a))


def _altura_da_linha(n_linhas: int) -> int:
    if n_linhas <= 1:
        return 420
    if n_linhas <= 3:
        return 260
    return 140


def montar_painel(resultados: list[tuple[str, list[Path]]], destino: Path) -> None:
    """Cada linha é um vídeo. A legenda fica acima, abaixo, ou some se for None."""
    if LEGENDA not in ("acima", "abaixo", None):
        raise SystemExit('LEGENDA precisa ser "acima", "abaixo" ou None')

    linhas = resultados if N_LINHAS is None else resultados[:N_LINHAS]
    if not linhas:
        raise SystemExit("nenhuma linha para o painel")

    altura = _altura_da_linha(len(linhas))
    margem = 16
    intervalo = 10
    fonte = _fonte(28 if len(linhas) <= 2 else 18)

    amostras = []
    for nome, frames in linhas:
        thumbs = []
        for frame in frames:
            img = recortar_centro(Image.open(frame).convert("RGB"), ZOOM_CENTRO)
            largura = max(1, round(img.width * altura / img.height))
            thumbs.append(img.resize((largura, altura), Image.Resampling.LANCZOS))
        amostras.append((nome, thumbs))

    largura_thumb = max(t.width for _, thumbs in amostras for t in thumbs)
    faixa = N_FRAMES * largura_thumb + (N_FRAMES - 1) * intervalo
    largura = margem * 2 + faixa

    if LEGENDA is None:
        texto_h = 0
        folga_legenda = 0
    else:
        sonda = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        texto_h = max(
            sonda.textbbox((0, 0), nome, font=fonte)[3] for nome, _ in amostras
        )
        folga_legenda = 8
    bloco = altura + folga_legenda + texto_h
    altura_total = margem + len(amostras) * bloco + (len(amostras) - 1) * margem + margem

    painel = Image.new("RGB", (largura, altura_total), (248, 248, 248))
    desenho = ImageDraw.Draw(painel)

    y = margem
    for nome, thumbs in amostras:
        if LEGENDA == "acima":
            caixa = desenho.textbbox((0, 0), nome, font=fonte)
            texto_x = margem + (faixa - (caixa[2] - caixa[0])) // 2
            desenho.text((texto_x, y), nome, fill=(20, 20, 20), font=fonte)
            y_fotos = y + texto_h + folga_legenda
        else:
            y_fotos = y
            if LEGENDA == "abaixo":
                caixa = desenho.textbbox((0, 0), nome, font=fonte)
                texto_x = margem + (faixa - (caixa[2] - caixa[0])) // 2
                desenho.text(
                    (texto_x, y_fotos + altura + folga_legenda),
                    nome,
                    fill=(20, 20, 20),
                    font=fonte,
                )
        x = margem
        for thumb in thumbs:
            painel.paste(thumb, (x, y_fotos))
            x += largura_thumb + intervalo
        y += bloco + margem

    painel.save(destino)


def main() -> None:
    pares = listar_videos()
    if not pares:
        raise SystemExit("nenhum vídeo de exemplo encontrado")

    raiz = EVAL_DIR / PASTA_RESULTADO
    resultados = []
    for video, nome in pares:
        if not video.is_file():
            raise SystemExit(f"vídeo não encontrado: {video}")
        print(nome)
        salvos = extrair_frames(video, raiz / nome)
        resultados.append((nome, salvos))

    painel = raiz / "resultado.png"
    montar_painel(resultados, painel)
    mostradas = len(resultados) if N_LINHAS is None else min(N_LINHAS, len(resultados))
    print(
        f"\n{len(resultados)} vídeo(s), {N_FRAMES} frames a cada {PASSO}. "
        f"painel com {mostradas} linha(s), zoom {ZOOM_CENTRO}, legenda {LEGENDA or 'nenhuma'}."
    )
    print(f"painel: {painel}")


if __name__ == "__main__":
    main()
