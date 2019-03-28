# AUTOR: STALIN FRANCIS
# SCRIP PARA ESTIMAR CON BM3D LAS IMAGENES QUE SERVIRAN PARA EL ENTRENAMIENTO DE LA RED NEURONAL CONVOLUCIONAL
# ESTE SCRIPT TAMBIEN ELIMINA LOS ARCHIVOS CREADOS Y NO NECESITADOS Y CLASIFICA EL DIRECTORIOS LAS ESTIMACIONES POR CADA NIVEL DE RUIDO

IMAGES=(00003)
EXTS=(_90 _180 _270 _flip _flop)

NOISES=(20 40 60 80 100)

for IMG in ${IMAGES[@]} ; do
	for NOISE in ${NOISES[@]} ; do
	../../../../bm3d_src/BM3Ddenoising ${IMG}.png ${NOISE} ${IMG}_Noisy_${NOISE}.png ${IMG}_Basic_${NOISE}.png ${IMG}_Denoised_${NOISE}.png ${IMG}_Diff_${NOISE}.png ${IMG}_Bias_${NOISE}.png ${IMG}_DiffBias_${NOISE}.png 1 bior 0 dct 1 rgb

	rm  ${IMG}_Basic_${NOISE}.png ${IMG}_Diff_${NOISE}.png ${IMG}_Bias_${NOISE}.png ${IMG}_DiffBias_${NOISE}.png 
	if [ ! -d ${NOISE} ]; then
		mkdir ${NOISE}
		mkdir ${NOISE}/Noisy
		mkdir ${NOISE}/Noisy/Categoria1
		mkdir ${NOISE}/Denoised/Categoria1

	fi
	if [  -f  ${IMG}_Noisy_${NOISE}.png ]; then
		mv ${IMG}_Noisy_${NOISE}.png ${NOISE}/Noisy/Categoria1/
	fi
	if [  -f  ${IMG}_Denoised_${NOISE}.png ]; then
		mv ${IMG}_Denoised_${NOISE}.png ${NOISE}/Denoised/Categoria1/
	fi
	done	

done


for IMG in ${IMAGES[@]} ; do
	for EXT in ${EXTS[@]} ; do
		for NOISE in ${NOISES[@]} ; do
			../../../../bm3d_src/BM3Ddenoising ${IMG}${EXT}.png ${NOISE} ${IMG}${EXT}_Noisy_${NOISE}.png ${IMG}${EXT}_Basic_${NOISE}.png ${IMG}${EXT}_Denoised_${NOISE}.png ${IMG}${EXT}_Diff_${NOISE}.png ${IMG}${EXT}_Bias_${NOISE}.png ${IMG}${EXT}_DiffBias_${NOISE}.png 1 bior 0 dct 1 rgb

			rm ${IMG}${EXT}_Noisy_${NOISE}.png ${IMG}${EXT}_Basic_${NOISE}.png ${IMG}${EXT}_Diff_${NOISE}.png ${IMG}${EXT}_Bias_${NOISE}.png ${IMG}${EXT}_DiffBias_${NOISE}.png 

			if [  -f  ${IMG}${EXT}_Noisy_${NOISE}.png ]; then
				mv ${IMG}${EXT}_Noisy_${NOISE}.png ${NOISE}/
			fi
			if [  -f  ${IMG}${EXT}_Denoised_${NOISE}.png ]; then
				mv ${IMG}${EXT}_Denoised_${NOISE}.png ${NOISE}/
			fi
		done	
	done
done	

