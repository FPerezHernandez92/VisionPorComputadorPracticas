// Pegar en Path del sistema: ;C:\opencv\build\x86\vc12\bin

#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

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
void pintaMIROI(vector<Mat> vim, string windowname, int col=CV_8UC3) {
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
	imshow(windowname,fin);
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
Mat AniadirBordes(Mat img, int tam_masc, int modo_borde=0) {
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
Mat AplicaConvolucion(Mat img, Mat mask, int tam_masc){
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
		merge(bandas,sal);
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
Mat EjercicioATrabajo1(string nomb1, float sigma, int modo_borde, int color, int parte=3) {
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

int main(int argc, char* argv[]) {
	/************************** APARTADO A *******************************/
	string nomb1 = "imagenes/dog.bmp", nomb2 = "imagenes/lena.jpg";
	vector<Mat> imagenes_col, imagenes_neg;
	//Sigma
	float sigma = 10;
	//Modo de los bordes que se añaden: 0(Uniforme a ceros), 1(Reflejado)
	int modo_borde;
	//0 Escala de grises, 1 color
	int color;
	Mat out;
	modo_borde = 0; color = 1; sigma = 10;
	out = EjercicioATrabajo1(nomb1, sigma, modo_borde, color);
	imagenes_col.push_back(out);
	modo_borde = 1; color = 1; sigma = 10;
	out = EjercicioATrabajo1(nomb1, sigma, modo_borde, color);
	imagenes_col.push_back(out);
	modo_borde = 0; color = 1; sigma = 3;
	out = EjercicioATrabajo1(nomb1, sigma, modo_borde, color);
	imagenes_col.push_back(out);
	modo_borde = 0; color = 1; sigma = 1;
	out = EjercicioATrabajo1(nomb1, sigma, modo_borde, color);
	imagenes_col.push_back(out);


	modo_borde = 0; color = 0; sigma = 10;
	out = EjercicioATrabajo1(nomb2, sigma, modo_borde, color);
	imagenes_neg.push_back(out);
	modo_borde = 1; color = 0; sigma = 10;
	out = EjercicioATrabajo1(nomb2, sigma, modo_borde, color);
	imagenes_neg.push_back(out);
	modo_borde = 0; color = 0; sigma = 5;
	out = EjercicioATrabajo1(nomb2, sigma, modo_borde, color);
	imagenes_neg.push_back(out);
	modo_borde = 0; color = 0; sigma = 3;
	out = EjercicioATrabajo1(nomb2, sigma, modo_borde, color);
	imagenes_neg.push_back(out);

	cout << "Muestro imagenes convolucionadas" << endl;
	pintaMIROI(imagenes_col, "ROI color dog");
	pintaMIROI(imagenes_neg, "ROI negro lena", CV_8UC1);

	
	/************************** APARTADO B *******************************/
	string nomb11 = "imagenes/cat.bmp", nomb12 = "imagenes/dog.bmp";
	string nomb3 = "imagenes/motorcycle.bmp", nomb4 = "imagenes/bicycle.bmp";
	string nomb5 = "imagenes/einstein.bmp", nomb6 = "imagenes/marilyn.bmp";
	string nomb7 = "imagenes/fish.bmp", nomb8 = "imagenes/submarine.bmp";
	string nomb9 = "imagenes/bird.bmp", nomb10 = "imagenes/plane.bmp";
	vector<Mat> imagenes_hib1, imagenes_hib2, imagenes_hib3, imagenes_hib4, imagenes_hib5;

	Mat img11 = leeimagen(nomb11, 1); Mat img12 = leeimagen(nomb12, 1);
	Mat img3 = leeimagen(nomb3, 1); Mat img4 = leeimagen(nomb4, 1);
	Mat img5 = leeimagen(nomb5, 1); Mat img6 = leeimagen(nomb6, 1);
	Mat img7 = leeimagen(nomb7, 1); Mat img8 = leeimagen(nomb8, 1);
	Mat img9 = leeimagen(nomb9, 1); Mat img10 = leeimagen(nomb10, 1);
	int sigma1, sigma2;

	cout << "Muestro imagenes hibridas" << endl;
	sigma1 = 5; sigma2 = 9;
	out = CrearImagenHibrida(img11, img12, imagenes_hib1, sigma1, sigma2);
	imagenes_hib1.push_back(out);
	pintaMIROI(imagenes_hib1, "Hibrida cat-dog");
	sigma1 = 2; sigma2 = 7;
	out = CrearImagenHibrida(img3, img4, imagenes_hib2, sigma1, sigma2);
	imagenes_hib2.push_back(out);
	pintaMIROI(imagenes_hib2, "Hibrida motorcycle-bicycle");
	sigma1 = 2; sigma2 = 6;
	out = CrearImagenHibrida(img5, img6, imagenes_hib3, sigma1, sigma2);
	imagenes_hib3.push_back(out);
	pintaMIROI(imagenes_hib3, "Hibrida einstein-marilyn");
	sigma1 = 3; sigma2 = 7;
	out = CrearImagenHibrida(img7, img8, imagenes_hib4, sigma1, sigma2);
	imagenes_hib4.push_back(out);
	pintaMIROI(imagenes_hib4, "Hibrida fish-submarine");
	sigma1 = 4; sigma2 = 7;
	out = CrearImagenHibrida(img9, img10, imagenes_hib5, sigma1, sigma2);
	imagenes_hib5.push_back(out);
	pintaMIROI(imagenes_hib5, "Hibrida bird-plane");

	
	/************************** APARTADO C *******************************/
	string nomb13 = "imagenes/cat.bmp", nomb14 = "imagenes/dog.bmp";
	Mat img13 = leeimagen(nomb13, 1); Mat img14 = leeimagen(nomb14, 1);
	int sigmaa, sigmab;
	sigmaa = 5; sigmab = 9;
	vector<Mat> imagenes_hib6, imagenes_pir;
	Mat hibrida = CrearImagenHibrida(img13, img14, imagenes_hib6, sigmaa, sigmab);
	imagenes_pir.push_back(hibrida);
	PiramideGaussiana(imagenes_pir);
	pintaMIROI(imagenes_pir, "Piramide Gaussiana");


	return 0;
}