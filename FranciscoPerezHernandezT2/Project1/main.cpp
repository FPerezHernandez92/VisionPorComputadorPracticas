// Pegar en Path del sistema: ;C:\opencv\build\x86\vc12\bin

#include<opencv2/opencv.hpp>
#include<math.h>
using namespace std;
using namespace cv;

/****************************************** TRABAJO 1 **********************************************/

bool crearimagenesdesalida = false;

Mat leeimagen(string filename, int flagColor) {
	Mat res = imread(filename, flagColor);
	return res;
}

void pintaI(Mat im, string name = "Ventana") {
	namedWindow(name, im.channels());
	imshow(name, im);
	if (crearimagenesdesalida)
		imwrite("salida/"+name+".jpg",im);
	cvWaitKey();
	destroyWindow(name);
}
void pintaMI(vector<Mat> vim) {
	string windowname = "Ventana ";
	string auxwindowname = windowname;
	for (int i = 0; i < vim.size(); i++) {
		auxwindowname.operator+=(to_string(i + 1));
		namedWindow(auxwindowname, vim[i].channels());
		imshow(auxwindowname, vim[i]);
		auxwindowname = windowname;
	}
	cvWaitKey();
	destroyAllWindows();
}
void pintaMIROI(vector<Mat> vim, string windowname, int col = CV_8UC3) {
	Mat fin = vim[0];
	for (int i = 1; i < (vim.size()); i++) {
		Size sz1 = fin.size();
		Size sz2 = vim[i].size();
		Mat auxfin(sz1.height, sz1.width + sz2.width, col);
		Mat izq(auxfin, Rect(0, 0, sz1.width, sz1.height));
		fin.copyTo(izq);
		Mat der(auxfin, Rect(sz1.width, 0, sz2.width, sz2.height));
		vim[i].copyTo(der);
		fin = auxfin;
	}
	if (crearimagenesdesalida)
		imwrite("salida/"+windowname+".jpg",fin);
	imshow(windowname, fin);
	cvWaitKey();
	destroyAllWindows();
}

/* Calculo el tamaño que tendrá las máscara en función del valor de sigma */
int TamanioMascara(float sigma) {
	int res = sigma * 2;
	res = res + 1;
	return res;
}
/* Función para calcular el valor en función de sigma y x */
float CalculaFuncion(float sigma, int x) {
	float aux;
	aux = (x*x) / (sigma*sigma);
	aux = exp((-0.5)*aux);
	return aux;
}
/* Función para calcular la máscara en función de sigma */
void CalculaMascara(Mat &mask, float sigma, int tam_masc) {
	int fin_masc = tam_masc / 2;
	int ini_masc = -fin_masc;
	float suma = 0.0, aux;
	//Calculo de la función Gaussiana 1D
	for (int i = ini_masc; i <= fin_masc; i++) {
		aux = CalculaFuncion(sigma, i);
		suma += aux;
		mask.push_back(aux);
	}
	//Normalizo el kernel
	for (int i = 0; i < tam_masc; i++) {
		mask.at<float>(i) = mask.at<float>(i) / suma;
	}
}
/* Función para añadir bordes a una imagen */
Mat AniadirBordes(Mat img, int tam_masc, int modo_borde = 0) {
	Mat res;
	//Borde Uniforme a ceros
	if (modo_borde == 0) {
		//Creo una matriz de color negra con la mitad de la máscara
		Mat aux = Mat::zeros(img.rows, tam_masc, img.type());
		//Añado a la imagen original la matriz con los laterales negros
		hconcat(img, aux, res);
		hconcat(aux, res, res);
		//Añado a la imagen original la matriz con los bordes superior e inferior
		aux = Mat::zeros(tam_masc, res.cols, img.type());
		vconcat(res, aux, res);
		vconcat(aux, res, res);
	}
	//Borde Reflejado
	else if (modo_borde == 1) {
		Mat aux;
		//Extraigo filas y columnas de los laterales
		aux = img.rowRange(0, img.rows);
		aux = aux.colRange(0, tam_masc);
		hconcat(aux, img, res);
		aux = img.rowRange(0, img.rows);
		aux = aux.colRange(img.cols - tam_masc, img.cols);
		hconcat(res, aux, res);
		//Extraigo la parte de arriba y de abajo
		aux = res.rowRange(0, tam_masc);
		aux = aux.colRange(0, res.cols);
		vconcat(aux, res, res);
		aux = res.rowRange(img.rows - tam_masc, img.rows);
		aux = aux.colRange(0, res.cols);
		vconcat(res, aux, res);
	}
	else
		cout << "Error al introducir el parametro modo_borde" << endl;
	return res;
}
/* Funciones auxiliares para aplicar convolución 1D */
void Convolucion1DPrimera(Mat &img, Mat&sal, Mat mask, int tam_masc) {
	vector<Mat> auxfila;
	Mat aux;
	for (int i = 0; i < img.rows; i++) {
		auxfila.push_back(sal.row(i));
		for (int j = 0; j < img.cols && (j + tam_masc < img.cols); j++) {
			aux = auxfila[0].colRange(j, j + tam_masc);
			auxfila[0].at<float>(j + tam_masc / 2) = aux.dot(mask.t());
		}
		auxfila.pop_back();
	}
}
void Convolucion1DSegunda(Mat &img, Mat&sal, Mat mask, int tam_masc) {
	vector<Mat> auxfila;
	Mat aux;
	for (int i = 0; i < img.rows; i++) {
		auxfila.push_back(sal.col(i));
		for (int j = 0; j < img.rows && (j + tam_masc < img.rows); j++) {
			aux = auxfila[0].rowRange(j, j + tam_masc);
			auxfila[0].at<float>(j + tam_masc / 2) = aux.dot(mask);
		}
		auxfila.pop_back();
	}
}
void Convolucion1DPrimeraColor(Mat& img, Mat& sal, Mat mask, int tam_masc, vector<Mat> bandas) {
	vector<Mat> auxfila;
	Mat aux;
	for (int i = 0; i<img.rows; i++) {
		auxfila.push_back(bandas[0].row(i));
		auxfila.push_back(bandas[1].row(i));
		auxfila.push_back(bandas[2].row(i));
		for (int j = 0; j<img.cols && j + tam_masc<img.cols; j++) {
			aux = auxfila[0].colRange(j, j + tam_masc);
			auxfila[0].at<float>(j + tam_masc / 2) = aux.dot(mask.t());
			aux = auxfila[1].colRange(j, j + tam_masc);
			auxfila[1].at<float>(j + tam_masc / 2) = aux.dot(mask.t());
			aux = auxfila[2].colRange(j, j + tam_masc);
			auxfila[2].at<float>(j + tam_masc / 2) = aux.dot(mask.t());
		}
		auxfila.pop_back();
		auxfila.pop_back();
		auxfila.pop_back();
	}
}
void Convolucion1DSegundaColor(Mat& img, Mat& sal, Mat mask, int tam_masc, vector<Mat> bandas) {
	vector<Mat> auxfila;
	Mat aux;
	for (int i = 0; i<img.cols; i++) {
		auxfila.push_back(bandas[0].col(i));
		auxfila.push_back(bandas[1].col(i));
		auxfila.push_back(bandas[2].col(i));
		for (int j = 0; j<img.rows && j + tam_masc<img.rows; j++) {
			aux = auxfila[0].rowRange(j, j + tam_masc);
			auxfila[0].at<float>(j + tam_masc / 2) = aux.dot(mask);
			aux = auxfila[1].rowRange(j, j + tam_masc);
			auxfila[1].at<float>(j + tam_masc / 2) = aux.dot(mask);
			aux = auxfila[2].rowRange(j, j + tam_masc);
			auxfila[2].at<float>(j + tam_masc / 2) = aux.dot(mask);
		}
		auxfila.pop_back();
		auxfila.pop_back();
		auxfila.pop_back();
	}
}
/* Función para aplicar convolución */
Mat AplicaConvolucion(Mat img, Mat mask, int tam_masc) {
	//Imagen en escala de grises
	Mat sal;
	if (img.type() == 0) {
		img.convertTo(sal, CV_32F);
		Convolucion1DPrimera(img, sal, mask, tam_masc);
		Convolucion1DSegunda(img, sal, mask, tam_masc);
		sal.convertTo(sal, CV_8U);
	}
	//Imagen en color
	else {
		vector<Mat> bandas;

		split(img, bandas);
		bandas[0].convertTo(bandas[0], CV_32F);
		bandas[1].convertTo(bandas[1], CV_32F);
		bandas[2].convertTo(bandas[2], CV_32F);

		Convolucion1DPrimeraColor(img, sal, mask, tam_masc, bandas);
		Convolucion1DSegundaColor(img, sal, mask, tam_masc, bandas);

		bandas[0].convertTo(bandas[0], CV_8U);
		bandas[1].convertTo(bandas[1], CV_8U);
		bandas[2].convertTo(bandas[2], CV_8U);
		merge(bandas, sal);
	}
	return sal;
}
/*Función para quitar los bordes añadidos */
void QuitarBordes(Mat& img, int tam_masc) {
	img = img.colRange(tam_masc, img.cols - tam_masc);
	img = img.rowRange(tam_masc, img.rows - tam_masc);
}
/*Función que calcula la convolución directamente en 2D */
void my_imGaussConvol(Mat img, Mat& out, float sigma) {
	int tam_masc = TamanioMascara(sigma);
	Mat mask;
	CalculaMascara(mask, sigma, tam_masc);
	Mat sal1;
	sal1 = AniadirBordes(img, tam_masc);
	out = AplicaConvolucion(sal1, mask, tam_masc);
	QuitarBordes(out, tam_masc);
}
Mat EjercicioATrabajo1(string nomb1, float sigma, int modo_borde, int color, int parte = 3) {
	Mat out;
	//Leo las imágenes pasadas por parámetros:
	Mat img = leeimagen(nomb1, color);

	/************************PARTE 1****************************/
	//Calculo la máscara:
	if (parte == 2) {
		int tam_masc = TamanioMascara(sigma);
		Mat mask;
		CalculaMascara(mask, sigma, tam_masc);

		/************************PARTE 2****************************/
		Mat sal1;
		sal1 = AniadirBordes(img, tam_masc, modo_borde);
		Mat sal2;
		sal2 = AplicaConvolucion(sal1, mask, tam_masc);
		QuitarBordes(sal2, tam_masc);
		out = sal2;
	}

	/************************PARTE 3****************************/
	if (parte == 3) {
		my_imGaussConvol(img, out, sigma);
	}

	return out;
}

/* Función para crear imagenes híbrdias */
Mat CrearImagenHibrida(Mat img1, Mat img2, vector<Mat>& imagenes, int sigma1, int sigma2) {
	Mat alta_frec, baja_frec, res;

	//Aplicamos convolución para obtener las altas frecuencias
	my_imGaussConvol(img1, alta_frec, sigma1);
	img1.convertTo(img1, CV_8UC3);
	alta_frec.convertTo(alta_frec, CV_8UC3);
	//Sacamos solo las altas frecuencias
	alta_frec = img1 - alta_frec;

	//Aplicamos convolución para obtener las bajas frecuencias
	my_imGaussConvol(img2, baja_frec, sigma2);
	//La imagen híbrida será la suma de las altas frecuencias de una con
	//las bajas frecuencias de la otra
	res = alta_frec + baja_frec;

	imagenes.push_back(alta_frec);
	imagenes.push_back(baja_frec);
	return res;
}
/* Función para crear una pirámide Gaussiana */
void PiramideGaussiana(vector<Mat>& imagenes) {
	Mat aux1;
	Mat aux2;
	imagenes[0].copyTo(aux1);
	imagenes[0].copyTo(aux2);
	int niveles = 5;
	for (int i = 0; i < niveles; i++) {
		pyrDown(aux1, aux2, Size(aux2.cols / 2, aux2.rows / 2));
		aux2.copyTo(aux1);
		imagenes.push_back(aux2);
	}
}

/*****************************************************************************************/
/*********************************** TRABAJO 2 *******************************************/
/*****************************************************************************************/

Mat encontrarHomografia(vector<Point2d> vec1, vector<Point2d> vec2) {
	Mat matA;
	matA = Mat::zeros(2 * vec1.size(), 9, CV_64F);

	//formo la matriz A a partir de las parejas de puntos de vec1 y vec2
	int filamatriz = 0;
	if (vec1.size() >= 4) {
		if (vec1.size() == vec2.size()) {
			for (int i = 0; i < vec1.size(); i++) {
				matA.at<double>(filamatriz, 0) = vec1[i].x;
				matA.at<double>(filamatriz, 1) = vec1[i].y;
				matA.at<double>(filamatriz, 2) = 1;
				matA.at<double>(filamatriz, 6) = -1 * vec2[i].x * vec1[i].x;
				matA.at<double>(filamatriz, 7) = -1 * vec2[i].x * vec1[i].y;
				matA.at<double>(filamatriz, 8) = -1 * vec2[i].x;
				filamatriz++;
				matA.at<double>(filamatriz, 3) = vec1[i].x;
				matA.at<double>(filamatriz, 4) = vec1[i].y;
				matA.at<double>(filamatriz, 5) = 1;
				matA.at<double>(filamatriz, 6) = -1 * vec2[i].y * vec1[i].x;
				matA.at<double>(filamatriz, 7) = -1 * vec2[i].y * vec1[i].y;
				matA.at<double>(filamatriz, 8) = -1 * vec2[i].y;
				filamatriz++;
			}
		}
		else
			cout << "No tienen el mismo tamanio vec1 y vec2 (en encontrarHomografia)" << endl;
	}
	else
		cout << "No tienen el minimo de puntos (en encontrarHomografia)" << endl;

	//Con el método SVDecomp voy a buscar los valores propios
	Mat matW, matU, matVT;
	SVDecomp(matA, matW, matU, matVT);

	//Creo la matriz con la homografía
	Mat maty = Mat::zeros(3, 3, CV_64F);
	maty.at<double>(0, 0) = matVT.at<double>(8, 0);
	maty.at<double>(0, 1) = matVT.at<double>(8, 1);
	maty.at<double>(0, 2) = matVT.at<double>(8, 2);
	maty.at<double>(1, 0) = matVT.at<double>(8, 3);
	maty.at<double>(1, 1) = matVT.at<double>(8, 4);
	maty.at<double>(1, 2) = matVT.at<double>(8, 5);
	maty.at<double>(2, 0) = matVT.at<double>(8, 6);
	maty.at<double>(2, 1) = matVT.at<double>(8, 7);
	maty.at<double>(2, 2) = matVT.at<double>(8, 8);

	return maty;
}

void Ejercicio1Trabajo2(Mat img1, Mat img2){
	cout << "\nEjecutando Ejercicio 1 Trabajo 2" << endl;

	/******************EJERCICIO 1.1 **************/
	//Selecciono 10 puntos de la imagen de forma dispersa y los meto en un vector
	vector<Point2d> puntos_img1;
	puntos_img1.push_back(Point2d(163, 50)); //esq sup izq
	puntos_img1.push_back(Point2d(510, 22)); //esq sup der
	puntos_img1.push_back(Point2d(153, 409)); //esq inf izq
	puntos_img1.push_back(Point2d(518, 451)); //esq inf der
	puntos_img1.push_back(Point2d(155, 210)); //lat izq 4ºcua
	puntos_img1.push_back(Point2d(514, 206)); //lat der 4ºcua
	puntos_img1.push_back(Point2d(308, 45)); //lat sup 4ºcua 
	puntos_img1.push_back(Point2d(291, 431)); //lat inf 4ºcua
	puntos_img1.push_back(Point2d(404, 150)); //3º fil, 6ºcua
	puntos_img1.push_back(Point2d(467, 326)); //6º fil, 7ºcua

	//De la segunda imagen selecciono los mismos 10 puntos y los almaceno 
	vector<Point2d> puntos_img2;
	puntos_img2.push_back(Point2d(160, 30)); //esq sup izq
	puntos_img2.push_back(Point2d(497, 111)); //esq sup der
	puntos_img2.push_back(Point2d(99, 384)); //esq inf izq
	puntos_img2.push_back(Point2d(425, 430)); //esq inf der
	puntos_img2.push_back(Point2d(133, 183)); //lat izq 4ºcua
	puntos_img2.push_back(Point2d(468, 251)); //lat der 4ºcua
	puntos_img2.push_back(Point2d(311, 59)); //lat sup 4ºcua 
	puntos_img2.push_back(Point2d(244, 406)); //lat inf 4ºcua
	puntos_img2.push_back(Point2d(385, 184)); //3º fil, 6ºcua
	puntos_img2.push_back(Point2d(400, 338)); //6º fil, 7ºcua


	/******************EJERCICIO 1.2 **************/
	//Busco la homografía de ambos vectores de puntos tanto de la imagen 1 respecto
	//de la imagen 2 como de la 2 respecto de la 1
	Mat H1;
	H1 = encontrarHomografia(puntos_img1, puntos_img2);
	Mat H2;
	H2 = encontrarHomografia(puntos_img2, puntos_img1);


	/******************EJERCICIO 1.3 **************/
	//Hago las pruebas de aplicar la homografía a las imágenes
	Mat salida1;
	vector<Mat> imagenes_mi_metodo_disperso;
	warpPerspective(img1, salida1, H1, salida1.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes_mi_metodo_disperso.push_back(salida1);

	Mat salida2;
	warpPerspective(img2, salida2, H2, salida2.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes_mi_metodo_disperso.push_back(salida2);

	pintaMIROI(imagenes_mi_metodo_disperso, "1Puntos dispersos mi metodo");

	//Calculo la homografía con findHomography para comparar los resultados
	Mat HF1 = findHomography(puntos_img1, puntos_img2, CV_RANSAC);
	Mat HF2 = findHomography(puntos_img2, puntos_img1, CV_RANSAC);

	Mat salida1f;
	vector<Mat> imagenes_findHomography_disperso;
	warpPerspective(img1, salida1f, HF1, salida1f.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes_findHomography_disperso.push_back(salida1f);

	Mat salida2f;
	warpPerspective(img2, salida2f, HF2, salida2f.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes_findHomography_disperso.push_back(salida2f);

	pintaMIROI(imagenes_findHomography_disperso, "1Puntos dispersos findHomography");


	/******************EJERCICIO 1.4 **************/
	vector<Mat> imagenes_mi_metodo_concentrados;
	vector<Point2d> puntos_img3;
	puntos_img3.push_back(Point2d(157, 51));
	puntos_img3.push_back(Point2d(204, 45));
	puntos_img3.push_back(Point2d(250, 43));
	puntos_img3.push_back(Point2d(162, 106));
	puntos_img3.push_back(Point2d(205, 105));
	puntos_img3.push_back(Point2d(249, 97));
	puntos_img3.push_back(Point2d(161, 153));
	puntos_img3.push_back(Point2d(204, 153));
	puntos_img3.push_back(Point2d(250, 256));
	puntos_img3.push_back(Point2d(171, 67));

	vector<Point2d> puntos_img4;
	puntos_img4.push_back(Point2d(150, 18));
	puntos_img4.push_back(Point2d(206, 39));
	puntos_img4.push_back(Point2d(257, 54));
	puntos_img4.push_back(Point2d(149, 83));
	puntos_img4.push_back(Point2d(199, 91));
	puntos_img4.push_back(Point2d(244, 103));
	puntos_img4.push_back(Point2d(140, 131));
	puntos_img4.push_back(Point2d(191, 144));
	puntos_img4.push_back(Point2d(240, 152));
	puntos_img4.push_back(Point2d(165, 40));

	Mat H3;
	H3 = encontrarHomografia(puntos_img3, puntos_img4);
	Mat H4;
	H4 = encontrarHomografia(puntos_img4, puntos_img3);

	Mat salida3;
	warpPerspective(img1, salida3, H3, salida3.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes_mi_metodo_concentrados.push_back(salida3);

	Mat salida4;
	warpPerspective(img2, salida4, H4, salida4.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes_mi_metodo_concentrados.push_back(salida4);

	pintaMIROI(imagenes_mi_metodo_concentrados, "1Puntos concentrados mi metodo");

	vector<Mat> imagenes_findHomography_concentrado;
	Mat HF3 = findHomography(puntos_img3, puntos_img4, CV_RANSAC);
	Mat HF4 = findHomography(puntos_img4, puntos_img3, CV_RANSAC);

	Mat salida3f;
	warpPerspective(img1, salida3f, HF3, salida3f.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes_findHomography_concentrado.push_back(salida3f);

	Mat salida4f;
	warpPerspective(img2, salida4f, HF4, salida4f.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes_findHomography_concentrado.push_back(salida4f);

	pintaMIROI(imagenes_findHomography_concentrado, "1Puntos concentrados findHomography");

	cout << "Fin Ejecucion Ejercicio 1 Trabajo 2" << endl;
}

//EJERCICIO 2

void briskFuncion(Mat& img, vector<KeyPoint>& keypoints, Mat& brisk, Mat& descriptors, int nthresh=30, int noctaves=3, float npatternScale=1.0f){
	//Creamos el detector con los parámetros deseados
	Ptr<FeatureDetector> BRISKD = BRISK::create(nthresh, noctaves, npatternScale);
	//Hacemos la detección de los puntos clave en la imágen
	BRISKD->detect(img, keypoints);
	//Sacamos los descriptores
	BRISKD->compute(img, keypoints, descriptors);
	//Pintamos los keypoints en la imagen y los pasamos a brisk
	drawKeypoints(img, keypoints, brisk);
}

void orbFuncion(Mat& img, vector<KeyPoint>& keypoints, Mat& orb, Mat& descriptors, int nnfeatures = 500, float nscaleFactor = 1.2000000048F, int nnlevels = 8, int nedgeThreshold = 8, int nfirstLevel = 0, int nWTA_K = 2, int nscoreType = 0, int npatchSize = 31, int nfastThreshold = 20) {
	//Creamos el detector con los parámetros deseados
	Ptr<FeatureDetector> ORBD = ORB::create(nnfeatures, nscaleFactor, nnlevels, nedgeThreshold, nfirstLevel, nWTA_K, nscoreType, npatchSize, nfastThreshold);
	//Hacemos la detección de los puntos clave en la imágen
	ORBD->detect(img, keypoints);	
	//Sacamos los descriptores
	ORBD->compute(img, keypoints, descriptors);	
	//Pintamos los keypoints en la imagen y los pasamos a orb
	drawKeypoints(img, keypoints, orb);
}

void Ejercicio2Trabajo2(Mat img1, Mat img2, int detector, int thresh, int octaves, float patternScale, int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize, int fastThreshold){
	cout << "\nEjecutando Ejercicio 2 Trabajo 2" << endl;

	//Creo las variables
	Mat brisk1, brisk2, orb1, orb2;
	Mat descriptors1, descriptors2, descriptors3, descriptors4;
	vector<KeyPoint> keypoints1, keypoints2, keypoints3, keypoints4;
	vector<Mat> imagenes_brisk, imagenes_orb;

	if (detector == 0 || detector == 1){
		//Paso el detector BRISK
		briskFuncion(img1, keypoints1, brisk1, descriptors1, thresh, octaves, patternScale);
		imagenes_brisk.push_back(brisk1);
		briskFuncion(img2, keypoints2, brisk2, descriptors2, thresh, octaves, patternScale);
		imagenes_brisk.push_back(brisk2);
		//Pinto los resultados del detector BRISK
		cout << "BRISK:Se han encontrado " << keypoints1.size() << " keypoints y " << keypoints2.size() << " keypoints" << endl;
		pintaMIROI(imagenes_brisk, "2Detector BRISK");
	}

	if (detector == 0 || detector == 2){
		//Paso el detector ORB
		orbFuncion(img1, keypoints3, orb1, descriptors3, nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
		imagenes_orb.push_back(orb1);
		orbFuncion(img2, keypoints4, orb2, descriptors4, nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
		imagenes_orb.push_back(orb2);
		//Pinto los resultados del detector ORB
		cout << "ORB:Se han encontrado " << keypoints3.size() << " keypoints y " << keypoints4.size() << " keypoints" << endl;
		pintaMIROI(imagenes_orb, "2Detector ORB");
	}

	cout << "Fin Ejecucion Ejercicio 2 Trabajo 2" << endl;
}

//EJERCICIO 3

void PuntosEnCorrespondencia(Mat img1, Mat img2, Mat descriptors1, Mat descriptors2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, bool crossCheck, Mat& salida, vector<DMatch>& matchesbuenos, int factor_correspondencia=15){
	vector<DMatch> matches;

	//Realizamos la búsqueda de las correspondencias
	BFMatcher matcher = BFMatcher::BFMatcher(NORM_L2, crossCheck);
	matcher.match(descriptors1, descriptors2, matches);

	float min_dist;
	//Lo ordeno para que sea mas eficiente
	sort(matches.begin(), matches.end());
	min_dist = matches[0].distance;
	//Buscamos las mejores correspondencias
	for (int i = 0; i < matches.size(); i++){
		if (matches[i].distance <= factor_correspondencia * min_dist){
			matchesbuenos.push_back(matches[i]);
		}
		else {
			//en el caso de que no consiga un mínimo de 4 correspondencias if 
			if (matchesbuenos.size() < 4){
				//aumento el factor de correspondencia
				factor_correspondencia++;
				i--;
			}
			else {
				break;
			}
		}
	}
	cout << "He pasado de " << matches.size() << " correspondencias a " 
	<< matchesbuenos.size() << " Valor final de factor_correspondencia: " 
	<< factor_correspondencia << endl;

	//Pintamos los puntos de correspondencia
	drawMatches(img1, keypoints1, img2, keypoints2, matchesbuenos, salida);
}

void Ejercicio3Trabajo2(Mat img1, Mat img2, int detector, int factor_correspondencia, bool crossCheck = 1){
	cout << "\nEjecutando Ejercicio 3 Trabajo 2" << endl;

	//Variables
	Mat descriptor1, descriptor2;
	Mat empty;
	vector<KeyPoint> keypoints1, keypoints2;

	//Primero sacamos los descriptores
	if (detector == 0 || detector == 1){
		briskFuncion(img1, keypoints1, empty, descriptor1);
		briskFuncion(img2, keypoints2, empty, descriptor2);

		Mat img_salida;
		vector<DMatch> matchesbuenos;
		//Ahora llamaremos a nuestra función con las imagenes, descriptores, keypoints y el flag si queremos crossCheck
		PuntosEnCorrespondencia(img1, img2, descriptor1, descriptor2, keypoints1, keypoints2, crossCheck, img_salida, matchesbuenos, factor_correspondencia);
		pintaI(img_salida, "3BRISK");
	}
	if (detector == 0 || detector == 2){
		orbFuncion(img1, keypoints1, empty, descriptor1);
		orbFuncion(img2, keypoints2, empty, descriptor2);

		Mat img_salida;
		vector<DMatch> matchesbuenos;
		//Ahora llamaremos a nuestra función con las imagenes, descriptores, keypoints y el flag si queremos crossCheck
		PuntosEnCorrespondencia(img1, img2, descriptor1, descriptor2, keypoints1, keypoints2, crossCheck, img_salida, matchesbuenos, factor_correspondencia);
		pintaI(img_salida, "3ORB");
	}

	cout << "Fin Ejecucion Ejercicio 3 Trabajo 2" << endl;
}

//EJERCICIO 4
void crearMosaico(Mat& mosaico, Mat img1, int factor_correspondencia=15){
	vector<KeyPoint> keypoints1, keypoints2;
	Mat empty, descriptors1, descriptors2;

	//Primero tenemos que sacar los keypoints, para ello uso brisk
	briskFuncion(img1, keypoints1, empty, descriptors1);
	briskFuncion(mosaico, keypoints2, empty, descriptors2);

	Mat img_sal;
	vector<DMatch> matchesbuenos;
	//Sacamos los puntos en correspondencias
	PuntosEnCorrespondencia(img1, mosaico, descriptors1, descriptors2, keypoints1, keypoints2, 1, img_sal, matchesbuenos, factor_correspondencia);

	vector<Point2f> key1, key2;
	//Sacamos los keypoints de las correspondencias buenas
	cout << "Total de correspondencias buenas para las dos imagenes: " << matchesbuenos.size() << endl;
	for (int i = 0; i < matchesbuenos.size(); i++){
		key1.push_back(keypoints1[matchesbuenos[i].queryIdx].pt);
		key2.push_back(keypoints2[matchesbuenos[i].trainIdx].pt);
	}

	//Buscamos la homografía de las imágenes
	Mat H = findHomography(key1, key2, CV_RANSAC);
	//Montamos las imagenes
	warpPerspective(img1, mosaico, H, mosaico.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
}

void prepararMosaico(Mat img, Mat& mosaico, int mirows, int micols){
	//Creo el almacén del mosaico
	mosaico = Mat(mirows, micols, img.type());
	resize(mosaico, mosaico, Size(mirows, micols));

	//Coloco la imagen en el centro
	Mat H0 = (Mat_<double>(3, 3) << 1, 0, (mosaico.cols / 2) - (img.cols), 0, 1, (mosaico.rows/2) - (img.rows ), 0, 0, 1);
	warpPerspective(img, mosaico, H0, Size(mirows, micols));
}


Mat eliminarZonasNegras2(Mat img){
	Vec3b aux;
	Point punto;
	int au1, au2, au0;
	int primerafilasincolor = 0;
	bool filasincolor = false;
	for (int i = 0; i < img.rows - 1 && !filasincolor; i++){
		for (int j = 0; j < img.cols - 1 && !filasincolor; j++){
			punto.x = j; punto.y = i;
			aux = img.at<Vec3b>(punto);
			au0 = (int)aux.val[0];
			au1 = (int)aux.val[1];
			au2 = (int)aux.val[2];
			if (au0 > 10 || au1 > 10 || au2 > 10){
				primerafilasincolor = i;
				filasincolor = true;
			}
		}
	}
	Mat nimg = Mat(img.rows - primerafilasincolor + 3, img.cols, img.type());
	int ni = 0;
	Point punto2;
	for (int i = primerafilasincolor - 3; i < img.rows - 1; i++){
		ni++;
		for (int j = 0; j < nimg.cols - 1; j++){
			punto.x = j; punto.y = ni;
			punto2.x = j; punto2.y = i;
			nimg.at<Vec3b>(punto) = img.at<Vec3b>(punto2);
		}
	}

	//pintaI(nimg);

	int primeracolumnasincolor = 0;
	bool columnasincolor = false;
	for (int i = 0; i < nimg.cols - 1 && !columnasincolor; i++){
		for (int j = 0; j < nimg.rows && !columnasincolor; j++){
			punto.x = i; punto.y = j;
			aux = nimg.at<Vec3b>(punto);
			au0 = (int)aux.val[0];
			au1 = (int)aux.val[1];
			au2 = (int)aux.val[2];
			if (au0 > 10 || au1 > 10 || au2 > 10){
				primeracolumnasincolor = i;
				columnasincolor = true;
			}
		}
	}
	int nj = 0;

	if (primeracolumnasincolor == 0){
		return nimg;
	}
	else {

		Mat nnimg = Mat(nimg.rows, nimg.cols - primeracolumnasincolor + 3, nimg.type());
		for (int i = 0; i < nnimg.rows - 1; i++){
			for (int j = primeracolumnasincolor - 3; j < nimg.cols - 1; j++){
				punto.x = nj; punto.y = i;
				punto2.x = j; punto2.y = i;
				nnimg.at<Vec3b>(punto) = nimg.at<Vec3b>(punto2);

				nj++;
			}
			nj = 0;
		}


		return nnimg;
	}
}
Mat eliminarZonasNegras(Mat img){
	Vec3b aux;
	Point punto;
	int au1, au2, au0;
	int ultimafilasincolor = img.rows - 1;
	bool filasincolor = false;
	for (int i = img.rows - 1; i >= 0 && !filasincolor; i--){
		for (int j = img.cols - 1; j>=0 && !filasincolor; j--){
			punto.x = j; punto.y = i;
			aux = img.at<Vec3b>(punto);
			au0 = (int)aux.val[0];
			au1 = (int)aux.val[1];
			au2 = (int)aux.val[2];
			if (au0 > 10 || au1 > 10 || au2 > 10){
				ultimafilasincolor = i;
				filasincolor = true;
			}
		}
	}
	Mat nimg = Mat(ultimafilasincolor+3, img.cols, img.type());
	for (int i = 0; i < nimg.rows; i++){
		for (int j = 0; j < nimg.cols; j++){
			punto.x = j; punto.y = i;
			nimg.at<Vec3b>(punto) = img.at<Vec3b>(punto);
		}
	}

	int ultimacolumnasincolor = nimg.cols - 1;
	bool columnasincolor = false;
	for (int i = nimg.cols - 1; i >= 0 && !columnasincolor; i--){
		for (int j = nimg.rows - 1; j >= 0 && !columnasincolor; j--){
			punto.x = i; punto.y = j;
			aux = nimg.at<Vec3b>(punto);
			au0 = (int)aux.val[0];
			au1 = (int)aux.val[1];
			au2 = (int)aux.val[2];
			if (au0 > 10 || au1 > 10 || au2 > 10){
				ultimacolumnasincolor = i;
				columnasincolor = true;
			}
		}
	}
	Mat nnimg = Mat(nimg.rows, ultimacolumnasincolor+3, nimg.type());
	for (int i = 0; i < nnimg.rows; i++){
		for (int j = 0; j < nnimg.cols; j++){
			punto.x = j; punto.y = i;
			nnimg.at<Vec3b>(punto) = nimg.at<Vec3b>(punto);
		}
	}
	nnimg = eliminarZonasNegras2(nnimg);
	return nnimg;
}


void Ejercicio4Trabajo2(Mat img1, Mat img2, int rows_p, int cols_p, int factor_correspondencia){
	cout << "\nEjecutando Ejercicio 4 Trabajo 2" << endl;

	Mat mosaico;
	//Preparo el mosaico
	prepararMosaico(img2, mosaico, rows_p, cols_p);
	//Creo el mosaico
	crearMosaico(mosaico, img1, factor_correspondencia);
	//Pinto el resultado
	pintaI(mosaico, "4Mosaico");



	mosaico = eliminarZonasNegras(mosaico);
	pintaI(mosaico, "4Mosaico Recortado");

	cout << "Fin Ejecucion Ejercicio 4 Trabajo 2" << endl;
}

//EJERCICIO 5

void crearSuperMosaico(vector<Mat> imagenes, Mat& mosaico_res, int mirows, int micols, int factor_correspondencia){
	cout << "En total hay " << imagenes.size() << " imagenes." << endl;
	
	Mat mosaico;
	int indice_comienzo = (imagenes.size()-1)/2;
	prepararMosaico(imagenes[indice_comienzo],mosaico,mirows,micols);
	cout << "Imagen " << indice_comienzo+1 << " aniadida." << endl;

	//Comienzo el mosaico del centro a la izquierda
	for (int i = indice_comienzo -1; i >= 0; i--){
		Mat img1 = imagenes[i];
		crearMosaico(mosaico, img1, factor_correspondencia);
		cout << "Imagen " << i+1 << " aniadida." << endl;
	}
	//Sigo el mosaico del centro a la derecha
	for (int i = indice_comienzo +1 ; i < imagenes.size(); i++){
		Mat img1 = imagenes[i];
		crearMosaico(mosaico, img1, factor_correspondencia);
		cout << "Imagen " << i+1 << " aniadida." << endl;
	}
	//Guardo el resultado en lo que devolveré 
	mosaico_res = mosaico;
}

void Ejercicio5Trabajo2(vector<Mat> imagenes, int rows_patanalla, int cols_pantalla, int factor_correspondencia){
	cout << "\nEjecutando Ejercicio 5 Trabajo 2" << endl;

	//Muestro las imágenes que van a participar en el mosaico
	if (imagenes.size() > 4){
		vector<Mat> parte1,parte2;
		for (int i=0;i<imagenes.size()/2;i++)
			parte1.push_back(imagenes[i]);
		for (int i=imagenes.size()/2;i<imagenes.size();i++)
			parte2.push_back(imagenes[i]);
		pintaMIROI(parte1, "5Primera parte imagenes del mosaico");
		pintaMIROI(parte2, "5Segunda parte imagenes del mosaico");
	}
	else 
		pintaMIROI(imagenes, "5Imagenes del mosaico");

	//Comienzo la creación del mosaico
	Mat mosaico;
	crearSuperMosaico(imagenes, mosaico, rows_patanalla, cols_pantalla, factor_correspondencia);
	pintaI(mosaico,"5Mosaico");
	mosaico = eliminarZonasNegras(mosaico);
	pintaI(mosaico, "5MosaicoRecortado");

	cout << "Fin Ejecucion Ejercicio 5 Trabajo 2" << endl;
}


int main(int argc, char* argv[]) {
	cout << endl;

	//Lectura de todas las imágenes usadas en el trabajo 2
	Mat img_tablero1 = leeimagen("imagenes/tablero1.jpg", 1);
	Mat img_tablero2 = leeimagen("imagenes/tablero2.jpg", 1);
	vector<Mat> imagenes_tableros;
	imagenes_tableros.push_back(img_tablero1);
	imagenes_tableros.push_back(img_tablero2);

	Mat img_mosaico1 = leeimagen("imagenes/mosaico002.jpg", 1);
	Mat img_mosaico2 = leeimagen("imagenes/mosaico003.jpg", 1);
	Mat img_mosaico3 = leeimagen("imagenes/mosaico004.jpg", 1);
	Mat img_mosaico4 = leeimagen("imagenes/mosaico005.jpg", 1);
	Mat img_mosaico5 = leeimagen("imagenes/mosaico006.jpg", 1);
	Mat img_mosaico6 = leeimagen("imagenes/mosaico007.jpg", 1);
	Mat img_mosaico7 = leeimagen("imagenes/mosaico008.jpg", 1);
	Mat img_mosaico8 = leeimagen("imagenes/mosaico009.jpg", 1);
	Mat img_mosaico9 = leeimagen("imagenes/mosaico010.jpg", 1);
	Mat img_mosaico10 = leeimagen("imagenes/mosaico011.jpg", 1);
	vector<Mat> imagenes_mosaico;
	imagenes_mosaico.push_back(img_mosaico1);
	imagenes_mosaico.push_back(img_mosaico2);
	imagenes_mosaico.push_back(img_mosaico3);
	imagenes_mosaico.push_back(img_mosaico4);
	imagenes_mosaico.push_back(img_mosaico5);
	imagenes_mosaico.push_back(img_mosaico6);
	imagenes_mosaico.push_back(img_mosaico7);
	imagenes_mosaico.push_back(img_mosaico8);
	imagenes_mosaico.push_back(img_mosaico9);
	imagenes_mosaico.push_back(img_mosaico10);

	Mat img_yosemite1 = leeimagen("imagenes/Yosemite1.jpg", 1);
	Mat img_yosemite2 = leeimagen("imagenes/Yosemite2.jpg", 1);
	Mat img_yosemite3 = leeimagen("imagenes/Yosemite3.jpg", 1);
	Mat img_yosemite4 = leeimagen("imagenes/Yosemite4.jpg", 1);
	resize(img_yosemite1, img_yosemite1, Size(320 + 160, 240 + 120));
	resize(img_yosemite2, img_yosemite2, Size(320 + 160, 240 + 120));
	resize(img_yosemite3, img_yosemite3, Size(320 + 160, 240 + 120));
	resize(img_yosemite4, img_yosemite4, Size(320 + 160, 240 + 120));
	/*resize(img_yosemite1, img_yosemite1, Size(320 , 240 ));
	resize(img_yosemite2, img_yosemite2, Size(320 , 240 ));
	resize(img_yosemite3, img_yosemite3, Size(320 , 240 ));
	resize(img_yosemite4, img_yosemite4, Size(320 , 240 ));*/
	vector<Mat> imagenes_yosemite;
	imagenes_yosemite.push_back(img_yosemite1);
	imagenes_yosemite.push_back(img_yosemite2);
	imagenes_yosemite.push_back(img_yosemite3);
	imagenes_yosemite.push_back(img_yosemite4);

	/*************************************************/
	/********************* EJERCICIO 1 ***************/
	/*************************************************/
	
	Ejercicio1Trabajo2(imagenes_tableros[0], imagenes_tableros[1]);		


	/*************************************************/
	/********************* EJERCICIO 2 ***************/
	/*************************************************/

	//Parámetros para el detector BRISK (por defecto)
	int thresh = 30;
	int octaves = 3;
	float patternScale = 1.0f;

	//Parámetros para el detector ORB (por defecto)
	int nfeatures = 500;
	float scaleFactor = 1.2000000048F; 
	int nlevels = 8, edgeThreshold = 8, firstLevel = 0, WTA_K = 2, scoreType = 0;
	int patchSize = 31, fastThreshold = 20;

	//0 ambos, 1 BRISK, 2 ORB
	int detector = 0;
	Ejercicio2Trabajo2(imagenes_yosemite[0], imagenes_yosemite[1], 0, thresh, octaves, patternScale, nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
	Ejercicio2Trabajo2(imagenes_yosemite[0], imagenes_yosemite[1], 0, 20, 2, 0.5f, 700, 1.4000000048F, 10, 10, -1, 3, 1, 35, 25);
	Ejercicio2Trabajo2(imagenes_yosemite[0], imagenes_yosemite[1], 0, 40, 4, 1.5f, 300, 1.0000000048F, 6, 6, -2, 2, -1, 25, 15);


	/*************************************************/
	/********************* EJERCICIO 3 ***************/
	/*************************************************/

	//0 ambos, 1 BRISK, 2ORB
	detector = 0;
	//Factor de correspondencia, para hacer la selección de las correspondencias mejores
	int factor_correspondencia = 1;
	bool crossCheck = 1;
	Ejercicio3Trabajo2(imagenes_yosemite[0],imagenes_yosemite[1], detector, factor_correspondencia, crossCheck);
	factor_correspondencia = 10;
	crossCheck = 1;
	Ejercicio3Trabajo2(imagenes_yosemite[0],imagenes_yosemite[1], detector, factor_correspondencia, crossCheck);
	crossCheck = 0;
	factor_correspondencia = 1;
	Ejercicio3Trabajo2(imagenes_yosemite[0],imagenes_yosemite[1], detector, factor_correspondencia, crossCheck);
	factor_correspondencia = 10;
	crossCheck = 0;
	Ejercicio3Trabajo2(imagenes_yosemite[0],imagenes_yosemite[1], detector, factor_correspondencia, crossCheck);


	/*************************************************/
	/********************* EJERCICIO 4 ***************/
	/*************************************************/

	//Pongo las dimensiones de mi pantalla, para poder mostrar todas las imágenes completas
	int rows_patanalla = 1800, cols_pantalla = 950;
	factor_correspondencia = 1;
	Ejercicio4Trabajo2(imagenes_yosemite[0],imagenes_yosemite[1],rows_patanalla,cols_pantalla,factor_correspondencia);


	/*************************************************/
	/********************* EJERCICIO 5 ***************/
	/*************************************************/

	rows_patanalla = 1800, cols_pantalla = 950;
	factor_correspondencia=4;
	Ejercicio5Trabajo2(imagenes_mosaico,rows_patanalla,cols_pantalla,factor_correspondencia);
	Ejercicio5Trabajo2(imagenes_yosemite,rows_patanalla,cols_pantalla,factor_correspondencia);

	cout << endl;
	return 0;
}

