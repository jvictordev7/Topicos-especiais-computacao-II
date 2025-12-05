"""
=========================================================================================
						PROCESSADOR AUTOMÁTICO DE DATASET OCR
=========================================================================================
VERSÃO: 1.0
OBJETIVO: Criar um dataset de alta qualidade (128x128 pixels, limpo) para
		  treinar Inteligências Artificiais de reconhecimento de escrita manual (OCR).

-----------------------------------------------------------------------------------------
						COMO ESTE ROBÔ PENSA? 🧠
-----------------------------------------------------------------------------------------

Este script é dividido em duas fases principais para imitar o processo de
extração de dados por um cirurgião digital, mas com filtros de qualidade
extremamente rigorosos.

--- FASE 1: EXTRAÇÃO INTELIGENTE (O "CIRURGIÃO") ---
O foco é localizar a tabela na folha escaneada, entender a grade e recortar cada
célula.

1.  Correção de Rotação (Deskew):
	A primeira ação é "endireitar" a folha, calculando o ângulo de inclinação do
	texto e girando a imagem para que fique perfeitamente alinhada na horizontal.

2.  Reconstrução da Grade (Lógica Híbrida):
	O script tenta "ver" as linhas da tabela. Se a linha está falhada ou apagada,
	ele usa a "Lógica Relativa": ele calcula a posição da célula perdida olhando
	para a posição média dos vizinhos (células encontradas na mesma coluna ou na
	linha de cabeçalho logo acima) para adivinhar o local exato do corte.

3.  Isolamento do Caractere ("Guilhotina de Luz"):
	Após o recorte inicial da célula, ele usa o Histograma de Projeção (um
	"Raio-X" vertical) para cortar qualquer pedaço de tinta que veio junto das
	células vizinhas. Ele elege um "Herói" (o objeto de tinta mais central) e
	elimina os intrusos laterais.

--- FASE 2: LIMPEZA RIGOROSA (O "SEGURANÇA DE BALADA") ---
O foco é rejeitar qualquer recorte que não seja uma imagem perfeita para treino
de IA.

1.  Filtros de Rejeição (Tolerância Zero - V28):
	O recorte é descartado (jogado no lixo) se:
	- O caractere principal estiver tocando qualquer borda da imagem (corte
	  errado).
	- O objeto for muito pequeno (sujeira ou pingo de tinta) ou muito esticado
	  (linha solta).
	- Não houver tinta suficiente (célula vazia).

2.  Padronização Final:
	- O caractere principal que foi aprovado é centralizado.
	- É redimensionado para 128x128 pixels mantendo a proporção original.
	- É salvo com a configuração ideal: fundo branco puro e caractere preto puro.

-----------------------------------------------------------------------------------------
						ESTRUTURA DE PASTAS 📂
-----------------------------------------------------------------------------------------
Para que o script funcione, seu projeto deve ter esta organização:

/seu_projeto/
  |-- processador_ocr_v28.py  (Este script, o cérebro do processo)
  |
  |-- folhas_coleta/ (PASTA DE ENTRADA: Coloque aqui seus scans .jpg ou .png)
  |      |-- Amostra 01.jpg
  |      |-- Amostra 02.png...
  |
  |-- dataset_bruto/ (PASTA INTERMEDIÁRIA: O script cria. Guarda recortes antes do filtro rigoroso.)
  |
  |-- dataset_limpo/ (PASTA DE SAÍDA: O script cria. Contém APENAS as imagens prontas para treinar a IA.)
  |      |-- A/
  |      |-- B/
  |      |-- 0/
  |      |-- 1/ ... etc (Uma subpasta para cada classe de letra/número)
  |
  |-- debug_visual/ (PASTA DE VERIFICAÇÃO: O script cria. Desenha os retângulos de corte nas folhas originais. Útil para ajustes.)

⚠️ AVISO IMPORTANTE: INSPEÇÃO MANUAL RECOMENDADA
Para atingir 100% de qualidade em datasets de escrita manual, é fundamental
que você faça uma inspeção manual final na pasta `dataset_limpo/`.

=========================================================================================
"""

import cv2
import numpy as np
import os
import glob
import shutil

# --- CONFIGURAÇÕES GERAIS ---
PASTA_ENTRADA = "folhas_coleta"
PASTA_BRUTA = "dataset_bruto"
PASTA_FINAL = "dataset_limpo"
PASTA_DEBUG = "debug_visual"
SALVAR_DEBUG = True
TAMANHO_FINAL = (128, 128)


def criar_diretorios(caminho_base):
	"""
	Cria a estrutura de pastas para as classes (A-F, 0-5).
	"""
	classes = list("ABCDEF012345")

	if not os.path.exists(caminho_base):
		os.makedirs(caminho_base)

	for c in classes:
		path = os.path.join(caminho_base, c)
		if not os.path.exists(path):
			os.makedirs(path)


def desentortar_imagem(image):
	"""
	Detecta se a folha foi escaneada torta e corrige a rotação automaticamente.
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)

	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	coords = np.column_stack(np.where(thresh > 0))
	if coords.size == 0:
		return image

	angle = cv2.minAreaRect(coords)[-1]

	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle

	if abs(angle) > 5.0:
		return image

	if abs(angle) > 0.1:
		(h, w) = image.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	return image


def limpar_vizinhos_histograma(thresh):
	"""
	Remove ruídos laterais (restos de outras letras) usando Histograma de Projeção.
	"""
	h, w = thresh.shape
	proj_x = np.sum(thresh, axis=0)
	has_ink = proj_x > (2 * 255)

	segments = []
	in_segment = False
	start = 0

	for x in range(w):
		if has_ink[x]:
			if not in_segment:
				start = x
				in_segment = True
		else:
			if in_segment:
				segments.append((start, x))
				in_segment = False
	if in_segment:
		segments.append((start, w))

	if not segments:
		return thresh

	center_x = w // 2
	best_seg = None
	min_dist = float('inf')

	for seg in segments:
		seg_center = (seg[0] + seg[1]) // 2
		dist = abs(seg_center - center_x)
		if dist < min_dist:
			min_dist = dist
			best_seg = seg

	if min_dist > (w * 0.3):
		return thresh

	mask_keep = np.zeros_like(thresh)
	x1 = max(0, best_seg[0] - 2)
	x2 = min(w, best_seg[1] + 2)
	mask_keep[:, x1:x2] = thresh[:, x1:x2]

	return mask_keep


def processar_recorte_celula(roi_celula):
	"""
	Pega o recorte bruto da tabela, isola o caractere e centraliza.
	"""
	h, w = roi_celula.shape[:2]
	if h == 0 or w == 0:
		return None

	if w > h * 1.4:
		new_w = int(h * 1.1)
		start_x = (w - new_w) // 2
		roi_celula = roi_celula[:, start_x: start_x + new_w]
		h, w = roi_celula.shape[:2]

	gray = cv2.cvtColor(roi_celula, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	cv2.rectangle(thresh, (0, 0), (w, h), (0), thickness=10)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

	thresh_isolado = limpar_vizinhos_histograma(thresh)
	cnts, _ = cv2.findContours(thresh_isolado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	center_x = w // 2
	center_y = h // 2
	min_dist = 99999
	heroi_box = None

	for c in cnts:
		area = cv2.contourArea(c)
		if area < 30:
			continue
		x, y, wc, hc = cv2.boundingRect(c)
		cx, cy = x + wc // 2, y + hc // 2
		dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

		if dist < min_dist:
			min_dist = dist
			heroi_box = (x, y, wc, hc)

	if heroi_box is None:
		canvas = np.ones((TAMANHO_FINAL[1], TAMANHO_FINAL[0], 3), dtype="uint8") * 255
		return canvas

	x, y, cw, ch = heroi_box
	roi_letra = roi_celula[y:y + ch, x:x + cw]

	desired = int(min(TAMANHO_FINAL) * 0.75)
	scale = desired / max(ch, cw)
	new_w = int(cw * scale)
	new_h = int(ch * scale)

	if new_w <= 0 or new_h <= 0:
		return None

	resized = cv2.resize(roi_letra, (new_w, new_h), interpolation=cv2.INTER_AREA)
	canvas = np.ones((TAMANHO_FINAL[1], TAMANHO_FINAL[0], 3), dtype="uint8") * 255

	x_off = (TAMANHO_FINAL[0] - new_w) // 2
	y_off = (TAMANHO_FINAL[1] - new_h) // 2
	canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

	return canvas


def reconstruir_grade_inteligente(roi_table):
	"""
	Analisa a tabela recortada e determina onde está cada célula.
	Usa lógica visual (linhas vistas) e lógica matemática (posição estimada)
	para corrigir falhas de impressão na grade.
	"""
	h_tab, w_tab = roi_table.shape[:2]

	gray = cv2.cvtColor(roi_table, cv2.COLOR_BGR2GRAY)
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								   cv2.THRESH_BINARY_INV, 11, 2)

	h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
	v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

	h_lines = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel), np.ones((3, 1)), iterations=2)
	v_lines = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel), np.ones((1, 3)), iterations=2)

	mask_grid = cv2.add(h_lines, v_lines)
	cnts, _ = cv2.findContours(cv2.bitwise_not(mask_grid), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	grid_detectado = {}
	ref_w = w_tab / 6.0

	for c in cnts:
		if cv2.contourArea(c) < 1000:
			continue
		x, y, w, h = cv2.boundingRect(c)
		if w > ref_w * 1.5:
			continue

		cx, cy = x + w / 2, y + h / 2
		col_idx = int(cx / ref_w)
		if col_idx > 5:
			col_idx = 5

		row_centers = [0.08, 0.33, 0.58, 0.83]
		pct_y = cy / h_tab
		distances = [abs(pct_y - c) for c in row_centers]
		row_idx = distances.index(min(distances))

		grid_detectado[(row_idx, col_idx)] = (x, y, w, h)

	grid_final = {}
	avg_w = int(ref_w)

	all_data_h = [v[3] for k, v in grid_detectado.items() if k[0] in [1, 3]]
	avg_h_data = int(np.median(all_data_h) * 1.1) if all_data_h else int(h_tab * 0.33)

	for row in range(4):
		last_valid_x = 0
		for col in range(6):
			if (row, col) in grid_detectado:
				rect = grid_detectado[(row, col)]
				grid_final[(row, col)] = (rect, "VISUAL")
				last_valid_x = rect[0] + rect[2]
			else:
				metodo = "ESTIMADO"
				pred_x = last_valid_x if col > 0 else 0
				pred_w = avg_w

				if row in [1, 3]:
					row_header = row - 1
					if (row_header, col) in grid_detectado:
						rect_header = grid_detectado[(row_header, col)]
						pred_y = rect_header[1] + rect_header[3]
						pred_h = avg_h_data
						metodo = "ANCORA"
					else:
						pred_y = int(h_tab * (0.17 if row == 1 else 0.67))
						pred_h = avg_h_data
				else:
					pred_h = int(h_tab * 0.17)
					pred_y = int(h_tab * (0.0 if row == 0 else 0.5))

				grid_final[(row, col)] = ((pred_x, pred_y, pred_w, pred_h), metodo)
				last_valid_x = pred_x + pred_w

	return grid_final


def executar_extracao():
	print("\n=== FASE 1: EXTRAÇÃO (CORTE DE TABELAS) ===")

	if os.path.exists(PASTA_BRUTA):
		shutil.rmtree(PASTA_BRUTA)
	criar_diretorios(PASTA_BRUTA)
	if SALVAR_DEBUG:
		if os.path.exists(PASTA_DEBUG):
			shutil.rmtree(PASTA_DEBUG)
		os.makedirs(PASTA_DEBUG)

	arquivos = glob.glob(os.path.join(PASTA_ENTRADA, "*.*"))
	if not arquivos:
		print(f"[ERRO] Nenhuma imagem na pasta '{PASTA_ENTRADA}'.")
		return

	print(f"[Status] Processando {len(arquivos)} folhas...")

	for arq in arquivos:
		nome_arquivo = os.path.basename(arq)
		img = cv2.imread(arq)
		if img is None:
			continue

		img = desentortar_imagem(img)
		img_debug = img.copy() if SALVAR_DEBUG else None

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5, 5), 0)
		thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
		dilated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
		cnts_tables, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		tables = [c for c in cnts_tables if cv2.contourArea(c) > 40000]
		boundingBoxes = [cv2.boundingRect(c) for c in tables]
		if tables:
			(tables, boundingBoxes) = zip(*sorted(zip(tables, boundingBoxes), key=lambda b: b[1][1]))

		count_recortes = 0

		for idx_bloco, table_cnt in enumerate(tables):
			x_tab, y_tab, w_tab, h_tab = cv2.boundingRect(table_cnt)
			roi_table = img[y_tab:y_tab + h_tab, x_tab:x_tab + w_tab]

			grid_dict = reconstruir_grade_inteligente(roi_table)

			if SALVAR_DEBUG:
				for (r, c), (rect, metodo) in grid_dict.items():
					rx, ry, rw, rh = rect
					color = (0, 255, 0) if metodo == "VISUAL" else (0, 0, 255)
					cv2.rectangle(img_debug, (x_tab + rx, y_tab + ry), (x_tab + rx + rw, y_tab + ry + rh), color, 2)

			linhas_interesse = [(1, list("ABCDEF")), (3, list("012345"))]

			for row_idx, labels in linhas_interesse:
				for col_idx, label in enumerate(labels):
					rect_data, _ = grid_dict[(row_idx, col_idx)]
					x_c, y_c, w_c, h_c = rect_data

					margin = 4
					roi_final = roi_table[y_c + margin: y_c + h_c - margin, x_c + margin: x_c + w_c - margin]

					img_final = processar_recorte_celula(roi_final)

					if img_final is not None:
						nome_base = f"{label}_P{nome_arquivo.split('.')[0]}_B{idx_bloco}"
						caminho_salvar = os.path.join(PASTA_BRUTA, label, f"{nome_base}.jpg")

						c = 1
						while os.path.exists(caminho_salvar):
							caminho_salvar = os.path.join(PASTA_BRUTA, label, f"{nome_base}_{c}.jpg")
							c += 1
						cv2.imwrite(caminho_salvar, img_final)
						count_recortes += 1

		print(f"  -> Folha '{nome_arquivo}': {count_recortes} caracteres extraídos.")
		if SALVAR_DEBUG and img_debug is not None:
			cv2.imwrite(os.path.join(PASTA_DEBUG, f"DEBUG_{nome_arquivo}"), img_debug)


def limpar_e_salvar_imagem(caminho_in, caminho_out):
	"""
	Aplica filtros de qualidade para decidir se a imagem serve para treino.
	Rejeita: Imagens cortadas na borda, sujeira pequena, linhas soltas.
	"""
	img = cv2.imread(caminho_in)
	if img is None:
		return False

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	h, w = thresh.shape

	cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not cnts:
		return False

	center_x, center_y = w // 2, h // 2
	good_blobs = []
	reject_file = False

	for c in cnts:
		area = cv2.contourArea(c)
		if area < 60:
			continue
		x, y, wc, hc = cv2.boundingRect(c)

		toca_borda = (x <= 1) or (y <= 1) or (x + wc >= w - 1) or (y + hc >= h - 1)
		ratio = wc / float(hc) if hc != 0 else 0
		e_linha = (ratio > 5.0) or (ratio < 0.20)

		cx_blob = x + wc // 2
		cy_blob = y + hc // 2
		dist_centro = np.sqrt((cx_blob - center_x) ** 2 + (cy_blob - center_y) ** 2)
		eh_central = dist_centro < (w * 0.4)

		if toca_borda:
			if eh_central:
				reject_file = True
				break
			continue

		if e_linha:
			continue

		good_blobs.append(c)

	if reject_file or not good_blobs:
		return False

	mask_final = np.zeros_like(thresh)
	cv2.drawContours(mask_final, good_blobs, -1, (255), -1)

	final_points = cv2.findNonZero(mask_final)
	if final_points is None:
		return False

	x, y, w_b, h_b = cv2.boundingRect(final_points)
	if w_b < 15 or h_b < 15:
		return False

	roi = mask_final[y:y + h_b, x:x + w_b]

	canvas = np.ones((TAMANHO_FINAL[1], TAMANHO_FINAL[0]), dtype="uint8") * 255
	target_size = int(min(TAMANHO_FINAL) * 0.75)
	scale = target_size / max(h_b, w_b)
	new_w = int(w_b * scale)
	new_h = int(h_b * scale)

	roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
	roi_resized = cv2.bitwise_not(roi_resized)

	y_off = (TAMANHO_FINAL[1] - new_h) // 2
	x_off = (TAMANHO_FINAL[0] - new_w) // 2
	canvas[y_off:y_off + new_h, x_off:x_off + new_w] = roi_resized

	os.makedirs(os.path.dirname(caminho_out), exist_ok=True)
	cv2.imwrite(caminho_out, canvas)
	return True


def executar_limpeza():
	print("\n=== FASE 2: LIMPEZA E PADRONIZAÇÃO ===")

	if not os.path.exists(PASTA_BRUTA):
		print(f"[ERRO] Pasta '{PASTA_BRUTA}' não encontrada. Rode a extração primeiro.")
		return

	if os.path.exists(PASTA_FINAL):
		shutil.rmtree(PASTA_FINAL)
	criar_diretorios(PASTA_FINAL)

	total_lidos = 0
	total_aprovados = 0

	for root, dirs, files in os.walk(PASTA_BRUTA):
		for file in files:
			if file.lower().endswith(('.jpg', '.png')):
				total_lidos += 1
				path_in = os.path.join(root, file)
				rel_path = os.path.relpath(path_in, PASTA_BRUTA)
				path_out = os.path.join(PASTA_FINAL, rel_path)

				if limpar_e_salvar_imagem(path_in, path_out):
					total_aprovados += 1

	print(f"\n[Resumo Final]")
	print(f"Processadas: {total_lidos}")
	print(f"Aprovadas:   {total_aprovados}")
	print(f"Rejeitadas:  {total_lidos - total_aprovados} (Baixa qualidade)")
	print(f"Dataset salvo em: {PASTA_FINAL}")


if __name__ == "__main__":
	while True:
		print("\n" + "=" * 30)
		print("   GERADOR DE DATASET OCR")
		print("=" * 30)
		print("1. Extrair (Cortar Folhas)")
		print("2. Limpar (Filtrar Dataset)")
		print("3. Executar Tudo (1 e 2)")
		print("4. Sair")
		opcao = input("Opção: ")

		if opcao == "1":
			executar_extracao()
		elif opcao == "2":
			executar_limpeza()
		elif opcao == "3":
			executar_extracao()
			executar_limpeza()
		elif opcao == "4":
			break
		else:
			print("Opção inválida.")

