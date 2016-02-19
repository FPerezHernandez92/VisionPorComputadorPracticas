// Pegar en Path del sistema: ;C:\opencv\build\x86\vc12\bin

#include<opencv2/opencv.hpp>
#include<math.h>
using namespace std;
using namespace cv;

/****************************************** TRABAJO 1 **********************************************/

Mat leeimagen(string filename, int flagColor) {
	Mat res = imread(filename, flagColor);
	return res;
}

void pintaI(Mat im) {
	namedWindow("Ventana", im.channels());
	imshow("Ventana", im);
	cvWaitKey();
	destroyWindow("Ventana");
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
	matA = Mat::zeros(2*vec1.size(), 9, CV_64F);

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
			cout << "No tienen el mismo tamanio vec1 y vec2 (encontrarHomografia)" << endl;
	}
	else
		cout << "No tienen el minimo de puntos (encontrarHomografia)" << endl;
	
	Mat matW, matU, matVT;
	SVDecomp(matA, matW, matU, matVT);

	Mat maty = Mat::zeros(3,3,CV_64F);
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

void Ejercicio1Trabajo2(){
	Mat img_tablero1 = leeimagen("imagenes/tablero1.jpg", 1);
	Mat img_tablero2 = leeimagen("imagenes/tablero2.jpg", 1);

	/******************EJERCICIO 1.1 **************/
	vector<Point2d> puntos_img_tablero1;
	puntos_img_tablero1.push_back(Point2d(163, 50)); //esq sup izq
	puntos_img_tablero1.push_back(Point2d(510, 22)); //esq sup der
	puntos_img_tablero1.push_back(Point2d(153, 409)); //esq inf izq
	puntos_img_tablero1.push_back(Point2d(518, 451)); //esq inf der
	puntos_img_tablero1.push_back(Point2d(155, 210)); //lat izq 4ºcua
	puntos_img_tablero1.push_back(Point2d(514, 206)); //lat der 4ºcua
	puntos_img_tablero1.push_back(Point2d(308, 45)); //lat sup 4ºcua 
	puntos_img_tablero1.push_back(Point2d(291, 431)); //lat inf 4ºcua
	puntos_img_tablero1.push_back(Point2d(404, 150)); //3º fil, 6ºcua
	puntos_img_tablero1.push_back(Point2d(467, 326)); //6º fil, 7ºcua

	vector<Point2d> puntos_img_tablero2;
	puntos_img_tablero2.push_back(Point2d(160, 30)); //esq sup izq
	puntos_img_tablero2.push_back(Point2d(497, 111)); //esq sup der
	puntos_img_tablero2.push_back(Point2d(99, 384)); //esq inf izq
	puntos_img_tablero2.push_back(Point2d(425, 430)); //esq inf der
	puntos_img_tablero2.push_back(Point2d(133, 183)); //lat izq 4ºcua
	puntos_img_tablero2.push_back(Point2d(468, 251)); //lat der 4ºcua
	puntos_img_tablero2.push_back(Point2d(311, 59)); //lat sup 4ºcua 
	puntos_img_tablero2.push_back(Point2d(244, 406)); //lat inf 4ºcua
	puntos_img_tablero2.push_back(Point2d(385, 184)); //3º fil, 6ºcua
	puntos_img_tablero2.push_back(Point2d(400, 338)); //6º fil, 7ºcua

	
	/******************EJERCICIO 1.2 **************/
	Mat H1;
	H1 = encontrarHomografia(puntos_img_tablero1, puntos_img_tablero2);
	Mat H2;
	H2 = encontrarHomografia(puntos_img_tablero2, puntos_img_tablero1);
	

	/******************EJERCICIO 1.3 **************/
	Mat salida1;
	vector<Mat> imagenes;
	warpPerspective(img_tablero1, salida1, H1, salida1.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes.push_back(salida1);

	Mat salida2;
	warpPerspective(img_tablero2, salida2, H2, salida2.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes.push_back(salida2);

	pintaMIROI(imagenes, "Puntos dispersos");


	/******************EJERCICIO 1.4 **************/
	vector<Point2d> puntos_img_tablero3;
	puntos_img_tablero3.push_back(Point2d(157, 51));
	puntos_img_tablero3.push_back(Point2d(204, 45));
	puntos_img_tablero3.push_back(Point2d(250, 43));
	puntos_img_tablero3.push_back(Point2d(162, 106));
	puntos_img_tablero3.push_back(Point2d(205, 105));
	puntos_img_tablero3.push_back(Point2d(249, 97));
	puntos_img_tablero3.push_back(Point2d(161, 153));
	puntos_img_tablero3.push_back(Point2d(204, 153));
	puntos_img_tablero3.push_back(Point2d(250, 256));
	puntos_img_tablero3.push_back(Point2d(171, 67));

	vector<Point2d> puntos_img_tablero4;
	puntos_img_tablero4.push_back(Point2d(150, 18));
	puntos_img_tablero4.push_back(Point2d(206, 39));
	puntos_img_tablero4.push_back(Point2d(257, 54));
	puntos_img_tablero4.push_back(Point2d(149, 83));
	puntos_img_tablero4.push_back(Point2d(199, 91));
	puntos_img_tablero4.push_back(Point2d(244, 103));
	puntos_img_tablero4.push_back(Point2d(140, 131));
	puntos_img_tablero4.push_back(Point2d(191, 144));
	puntos_img_tablero4.push_back(Point2d(240, 152));
	puntos_img_tablero4.push_back(Point2d(165, 40));
	
	Mat H3;
	H3 = encontrarHomografia(puntos_img_tablero3, puntos_img_tablero4);
	Mat H4;
	H4 = encontrarHomografia(puntos_img_tablero4, puntos_img_tablero3);

	imagenes.clear();
	Mat salida3;
	warpPerspective(img_tablero1, salida3, H3, salida3.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes.push_back(salida3);

	Mat salida4;
	warpPerspective(img_tablero2, salida4, H4, salida4.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
	imagenes.push_back(salida4);

	pintaMIROI(imagenes, "Puntos concentrados");

}

//EJERCICIO 2

void briskFuncion(Mat& img, vector<KeyPoint>& keypoints, Mat& brisk, Mat& descriptors){
	int thresh = 30;
	int octaves = 3;
	float patternScale = 1.0f;

	Ptr<FeatureDetector> BRISKD=BRISK::create(thresh,octaves,patternScale);
	BRISKD->detect(img, keypoints);
	BRISKD->compute(img,keypoints,descriptors);
	drawKeypoints(img,keypoints,brisk);
}

void orbFuncion(Mat& img, vector<KeyPoint>& keypoints, Mat& orb, Mat& descriptors) {
	float scaleFactor = 1.2000000048F;
	int nfeatures = 500, nlevels = 8, edgeThreshold = 8, firstLevel = 0, WTA_K = 2, scoreType = 0, patchSize = 31, fastThreshold = 20;

	Ptr<FeatureDetector> ORBD = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
	ORBD->detect(img, keypoints);
	ORBD->compute(img,keypoints,descriptors);
	drawKeypoints(img,keypoints,orb);
}

void Ejercicio2Trabajo2(){
	Mat img_yosemite1 = leeimagen("imagenes/Yosemite1.jpg", 1);
	Mat img_yosemite2 = leeimagen("imagenes/Yosemite2.jpg", 1);

	Mat brisk1, brisk2, orb1, orb2;
	Mat descriptors1, descriptors2, descriptors3, descriptors4;
	vector<KeyPoint> keypoints1, keypoints2, keypoints3, keypoints4;
	vector<Mat> imagenes_brisk, imagenes_orb;

	briskFuncion(img_yosemite1, keypoints1, brisk1, descriptors1);
	imagenes_brisk.push_back(brisk1);
	briskFuncion(img_yosemite2, keypoints2, brisk2, descriptors2);
	imagenes_brisk.push_back(brisk2);
	pintaMIROI(imagenes_brisk, "Brisk");

	orbFuncion(img_yosemite1, keypoints3, orb1, descriptors3);
	imagenes_orb.push_back(orb1);
	orbFuncion(img_yosemite2, keypoints4, orb2, descriptors4);
	imagenes_orb.push_back(orb2);
	pintaMIROI(imagenes_orb, "Orb");
}

//EJERCICIO 3

void PuntosEnCorrespondencia(Mat img1, Mat img2, Mat descriptors1, Mat descriptors2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, bool crossCheck, Mat& salida, vector<DMatch>& matchesbuenos){
	vector<DMatch> matches;

	//Realizamos el matching
	BFMatcher matcher = BFMatcher::BFMatcher(NORM_L2, crossCheck);
	matcher.match(descriptors1, descriptors2, matches);

	float min_dist;
	//Buscamos el mejor matching
	sort(matches.begin(), matches.end());
	min_dist = matches[0].distance;
	for (int i = 0; i < matches.size(); i++){
		if (matches[i].distance <= 3 * min_dist)
			matchesbuenos.push_back(matches[i]);
		else
			break;
	}

	//Buscamos los puntos de correspondencia
	drawMatches(img1, keypoints1, img2, keypoints2, matches, salida);
}

void Ejercicio3Trabajo2(){
	Mat img_yosemite1 = leeimagen("imagenes/Yosemite1.jpg", 1);
	Mat img_yosemite2 = leeimagen("imagenes/Yosemite2.jpg", 1);

	Mat descriptor1, descriptor2;
	Mat empty;
	vector<KeyPoint> keypoints1, keypoints2;
	
	bool crossCheck = 1;

	//Primero scamos los descriptores
	orbFuncion(img_yosemite1, keypoints1, empty, descriptor1);
	orbFuncion(img_yosemite2, keypoints2, empty, descriptor2);

	//Ahora llamaremos a nuestra función con las imagenes, descriptores, keypoints y el flag si queremos crossCheck
	Mat img_salida;
	vector<DMatch> matchesbuenos;
	PuntosEnCorrespondencia(img_yosemite1, img_yosemite2, descriptor1, descriptor2, keypoints1, keypoints2, crossCheck, img_salida, matchesbuenos);
	pintaI(img_salida);
}

//EJERCICIO 4

void crearMosaico(Mat img1, Mat img2, Mat& mosaico){
	vector<KeyPoint> keypoints1, keypoints2;
	Mat empty, descriptors1, descriptors2;

	//Primero tenemos que sacar los keypoints, para ello uso brisk
	briskFuncion(img1, keypoints1, empty, descriptors1);
	briskFuncion(img2, keypoints2, empty, descriptors2);

	Mat img_sal;
	vector<DMatch> matchesbuenos;
	//Como en el ejercicio anterior sacamos los puntos en correspondencias
	PuntosEnCorrespondencia(img1, img2, descriptors1, descriptors2, keypoints1, keypoints2, 1, img_sal, matchesbuenos);

	Mat H;
	Mat H0 = (Mat_<double>(3, 3) << 1, 0, img2.cols / 4, 0, 1, img2.rows / 4, 0, 0, 1);
	warpPerspective(img2, mosaico, H0, Size(img1.rows + img2.rows, img2.cols +  img1.cols));

	vector<Point2f> imagen, escena;
	//Sacamos los keypoints de las correspondencias buenas
	for (int i = 0; i < matchesbuenos.size(); i++){
		imagen.push_back(keypoints1[matchesbuenos[i].queryIdx].pt);
		escena.push_back(keypoints2[matchesbuenos[i].trainIdx].pt);
	}

	//Buscamos la homografía de las imágenes
	H = findHomography(imagen, escena, CV_RANSAC);
	H = H*H0;
	//Montamos las imagenes
	warpPerspective(img1, mosaico, H, mosaico.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
}

void Ejercicio4Trabajo2(){
	Mat img_yosemite1 = leeimagen("imagenes/Yosemite1.jpg", 1);
	Mat img_yosemite2 = leeimagen("imagenes/Yosemite2.jpg", 1);

	Mat mosaico;
	crearMosaico(img_yosemite1, img_yosemite2, mosaico);

	pintaI(mosaico);
}

//EJERCICIO 5

void crearSuperMosaico(Mat img1, Mat img2, Mat img3, Mat img4, Mat img5, Mat& mosaico){
	Mat aux;
	crearMosaico(img1, img2, aux);
	crearMosaico(img3, aux, aux);
	crearMosaico(img4, aux, aux);
	//Mat miaux = aux.adjustROI(100, 200, 100, 200);
	//pintaI(miaux);
	//Mat miaux2 = aux.adjustROI(200, 300, 200, 300);
	//pintaI(miaux2);
	//crearMosaico(img5, miaux, aux);
	//aux = aux.adjustROI(200, 200, 200, 200);
	mosaico = aux;
}

void Ejercicio5Trabajo2(){
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
	Mat mosaico;

	crearSuperMosaico(img_mosaico1, img_mosaico2, img_mosaico3, img_mosaico4, img_mosaico5, mosaico);

	pintaI(mosaico);
	
}

int main(int argc, char* argv[]) {
	/*************************************************/
	/********************* EJERCICIO 1 ***************/
	/*************************************************/
	//Ejercicio1Trabajo2();		
	
	
	/*************************************************/
	/********************* EJERCICIO 2 ***************/
	/*************************************************/
	//Ejercicio2Trabajo2();


	/*************************************************/
	/********************* EJERCICIO 3 ***************/
	/*************************************************/
	//Ejercicio3Trabajo2();


	/*************************************************/
	/********************* EJERCICIO 4 ***************/
	/*************************************************/
	Ejercicio4Trabajo2();


	/*************************************************/
	/********************* EJERCICIO 5 ***************/
	/*************************************************/
	//Ejercicio5Trabajo2();



	return 0;
}

