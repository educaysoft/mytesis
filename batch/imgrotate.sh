# AUTOR: STALIN FRANCIS
# SCRIP PARA CREAR MÁS IMAGENES PARA ALIMENTAR LA RED NEURONAL ROTANDO LA IMAGEN ORIGINAL 90 180 Y 270 GRADOS ADEMAS REFLEJANDOLA
# HORIZONTAL Y VERTICALMENTE.

IMAGES=(00003)
ETS=(.png _90.png _180.png _270.png _horizontal.png _vertical.png)

for IMG in ${IMAGES[@]} ;  do
		convert  ${IMG}.png -rotate 90  ${IMG}_90.png
		convert  ${IMG}.png -rotate 180 ${IMG}_180.png
		convert  ${IMG}.png -rotate 270 ${IMG}_270.png
		convert  ${IMG}.png  -flip ${IMG}_fip.png
		convert  ${IMG}.png -flop ${IMG}_flop.png
done
