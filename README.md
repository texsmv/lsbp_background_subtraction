Local SVD Binary Pattern BackgroundSubstractor

Requerimientos:
Cuda Toolkit
Opencv 3.X(configurado con cuda)
libconfig( sudo apt-get install libconfig++8-dev)11

INstrucciones:
1.- Hacer: make
2.- Colocar los videos que se desean procesar en la carpeta "data/video input"
3.- Opcionalmente cambiar la configuracion en el archivo config.cfg
4.- Ejecutar ./exe hight width nombre_input nombre_output batch_size numero_frames
Ejm: ./exe 200 400 test4.mp4 anochecer 1 2000
- Si el numero de frames es igual a cero se procesara todo el video
Ejm ./exe 200 400 test4.mp4 anochecer 1 0
- Los videos se guardan en "data/video output"
