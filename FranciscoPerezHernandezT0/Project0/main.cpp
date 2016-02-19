// Pegar en Path del sistema: ;C:\opencv\build\x86\vc12\bin

#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void Ejercicio12Guion() {
	//Definir un objeto de tipo Mat y Leer la imagen en el. Usar la función imread. Muestra 
	//la imagen en una ventana usando namedWindow, imshow, keyWait, destroyWindow.
	Mat img = imread("imagenes/lena.jpg", 1);
	namedWindow("ventana", 1);
	imshow("ventana", img);
	cvWaitKey(0);
	destroyWindow("ventana");
}
void Ejercicio13Guion() {
	//Hacer lo anterior tanto con imágenes de niveles de gris como de color.Transformar una 
	//imagen de color en una de niveles de gris con cvtColor.Visualizar ambos resultados.
	Mat img1 = imread("imagenes/lena.jpg", 1);
	Mat img2 = img1; 
	cvtColor(img2, img2, CV_BGR2GRAY);
	namedWindow("ventana1", 1);
	namedWindow("ventana2", 1);
	imshow("ventana1", img1);
	imshow("ventana2", img2);
	cvWaitKey(0);
	destroyWindow("ventana1");
	destroyWindow("ventana2");
}
void Ejercicio14Guion() {
	//Imprimir por pantalla los valores de la cabecera de las imágenes: rows, cols, channels, type. 
	//Hacerlo con imágenes en color y de niveles de gris. Analizar las diferencias.
	Mat img1 = imread("imagenes/lena.jpg", 1);
	Mat img2 = img1;
	cvtColor(img2, img2, CV_BGR2GRAY);
	Mat img3 = imread("imagenes/lena.jpg", 0);

	cout << "Color rows: " << img1.rows << endl;
	cout << "Convertida rows: " << img2.rows << endl;
	cout << "Gris rows: " << img3.rows << endl;
	cout << "Color cols: " << img1.cols << endl;
	cout << "Convertida cols: " << img2.cols << endl;
	cout << "Gris cols: " << img3.cols << endl;
	cout << "Color channels: " << img1.channels() << endl;
	cout << "Convertida channels: " << img2.channels() << endl;
	cout << "Gris channels: " << img3.channels() << endl;
	cout << "Color type: " << img1.type() << endl;
	cout << "Convertida type: " << img2.type() << endl;
	cout << "Gris type: " << img3.type() << endl;

	namedWindow("ventana1", 1);
	namedWindow("ventana2", 1);
	namedWindow("ventana3", 1);
	imshow("ventana1", img1);
	imshow("ventana2", img2);
	imshow("ventana3", img2);
	cvWaitKey(0);
	destroyWindow("ventana1");
	destroyWindow("ventana2");
	destroyWindow("ventana3");
}
void Ejercicio15Guion() {
	//Buscar el concepto de ROI de una imagen en la documentación y usarlo para mostrar 
	//dos imágenes distintas en una misma imagen de salida.
	Mat img1 = imread("imagenes/cat.bmp", 1);
	Mat img2 = imread("imagenes/lena.jpg", 1);
	Size sz1 = img1.size();
	Size sz2 = img2.size();
	Mat img3(sz1.height, sz1.width + sz2.width, CV_8UC3);
	Mat izq(img3, Rect(0, 0, sz1.width, sz1.height));
	img1.copyTo(izq);
	Mat der(img3, Rect(sz1.width, 0, sz2.width, sz2.height));
	img2.copyTo(der);

	imshow("ventana1", img1);
	imshow("ventana2", img2);
	imshow("ventana3", img3);

	cvWaitKey(0);

	destroyWindow("ventana1");
	destroyWindow("ventana2");
	destroyWindow("ventana3");
}
void Ejercicio16Guion() {
	//Ahora vamos a añadirle nueve puntos que dividan la imagen en 16 sectores imaginarios iguales, 
	//como se muestra a continuación.
	Mat img = imread("imagenes/lena.jpg", 1);
	Point pixel;
	Vec3b punto;
	punto.val[0] = 255; punto.val[1] = 255; punto.val[2] = 255;

	int x = img.cols / 4, y = img.rows / 4;
	for (int i = 1; i <= 3; i++) {
		pixel.x = x * i; pixel.y = y;
		img.at<Vec3b>(pixel) = punto;
		pixel.x = x * i; pixel.y = y*2;
		img.at<Vec3b>(pixel) = punto;
		pixel.x = x * i; pixel.y = y*3;
		img.at<Vec3b>(pixel) = punto;
	}
	
	namedWindow("ventana", 1);
	imshow("ventana", img);
	cvWaitKey();
	destroyWindow("ventana");
}
void pintarCruz(Mat img, vector<Point> vec) {
	Vec3b punto;
	punto.val[0] = 255;
	punto.val[1] = 0;
	punto.val[2] = 0;
	for (int i = 0; i < vec.size(); i++) {
		img.at<Vec3b>(vec[i]) = punto;
		Point nuevo; 
		nuevo.x = vec[i].x + 1; nuevo.y = vec[i].y + 0;
		img.at<Vec3b>(nuevo) = punto;
		nuevo.x = vec[i].x + 2; nuevo.y = vec[i].y + 0;
		img.at<Vec3b>(nuevo) = punto;
		nuevo.x = vec[i].x - 1; nuevo.y = vec[i].y + 0;
		img.at<Vec3b>(nuevo) = punto;
		nuevo.x = vec[i].x - 2; nuevo.y = vec[i].y + 0;
		img.at<Vec3b>(nuevo) = punto;
		nuevo.x = vec[i].x + 0; nuevo.y = vec[i].y + 1;
		img.at<Vec3b>(nuevo) = punto;
		nuevo.x = vec[i].x + 0; nuevo.y = vec[i].y + 2;
		img.at<Vec3b>(nuevo) = punto;
		nuevo.x = vec[i].x + 0; nuevo.y = vec[i].y - 1;
		img.at<Vec3b>(nuevo) = punto;
		nuevo.x = vec[i].x + 0; nuevo.y = vec[i].y - 2;
		img.at<Vec3b>(nuevo) = punto;
	}
}
void Ejercicio2Guion() {
	//Escribir unan función que tenga como entrada una imagen de grises o de color, y un vector vec con 
	//coordenadas de puntos ( vector<Point> vec, tipo Point de OpenCV). Pintar una cruz centrada en cada 
	//uno de los elementos del vector de entrada con longitud de brazo de 2 pixeles. ( color a elegir). 
	//Visualizar la imagen final.
	Mat img = imread("imagenes/lena.jpg", 1);
	vector<Point> vec;
	Point punto;
	punto.x = 50; punto.y = 50;
	vec.push_back(punto);
	punto.x = 100; punto.y = 100;
	vec.push_back(punto);
	punto.x = 150; punto.y = 150;
	vec.push_back(punto);
	pintarCruz(img, vec);
	namedWindow("ventana", 1);
	imshow("ventana", img);
	cvWaitKey(0);
	destroyWindow("ventana");
}

Mat leeimagen(string filename, int flagColor) {
	Mat res = imread(filename, flagColor);
	return res;
}
void Ejercicio1Trabajo0() {
	//Escribir una función que lea una imagen en niveles de gris o en color(im = leeimagen(filename, flagColor))
	Mat img;
	img = leeimagen("imagenes/lena.jpg", 1);
}
void pintaI(Mat im) {
	namedWindow("Ventana", im.channels());
	imshow("Ventana", im);
	cvWaitKey();
	destroyWindow("Ventana");
}
void Ejercicio2Trabajo0() {
	//Escribir una función que visualice una imagen (pintaI(im))
	Mat img;
	img = leeimagen("imagenes/lena.jpg", 1);
	pintaI(img);
}
void pintaMI(vector<Mat> vim) {
	cout << vim.size() << endl;
	string windowname = "Ventana ";
	string auxwindowname = windowname;
	for (int i = 0; i < vim.size(); i++) {
		auxwindowname.operator+=(to_string(i + 1));
		namedWindow(auxwindowname, vim[i].channels());
		imshow(auxwindowname, vim[i]);
		auxwindowname = windowname;
	}
	cvWaitKey();
	/*for (int i = 0; i < vim.size(); i++) {
	auxwindowname.operator+=(to_string(i+1));
	destroyWindow(auxwindowname);
	auxwindowname = windowname;
	}*/
	destroyAllWindows();
}
void Ejercicio3Trabajo0() {
	//Escribir una función que visualice varias imágenes a la vez: pintaMI(vim). (vim será una secuencia de imágenes) 
	//¿Qué pasa si las imágenes no son todas del mismo tipo : (nivel de gris, color, blanco - negro) ?
	//Se visualizan utilizanco cada canal que tiene la imagen definida cuando se ha cargado.
	Mat img1 = leeimagen("imagenes/lena.jpg", 1);
	Mat img2 = leeimagen("imagenes/lena.jpg", 0);
	vector<Mat> vim;

	vim.push_back(img1);
	vim.push_back(img2);

	pintaMI(vim);
}
void modificapixeles(Mat img, vector<Point> pixeles) {
	Vec3b punto;
	punto.val[0] = 255;
	punto.val[1] = 0;
	punto.val[2] = 0;
	for (int i = 0; i < pixeles.size(); i++) {
		img.at<Vec3b>(pixeles[i]) = punto;
	}
}
void Ejercicio4Trabajo0() {
	//Escribir una función que modifique el valor en una imagen de una lista de coordenadas de píxeles.
	Mat img = leeimagen("imagenes/lena.jpg", 1);
	vector<Point> pixeles;

	Point pixel = (10, 10);
	pixeles.push_back(pixel);
	pixel.x = 20;
	pixel.y = 20;
	pixeles.push_back(pixel);
	pixel.x = 50;
	pixel.y = 50;
	pixeles.push_back(pixel);

	modificapixeles(img, pixeles);

	pintaI(img);
}

int main(int argc, char* argv[]) {
	Ejercicio12Guion();
	Ejercicio13Guion();
	Ejercicio14Guion();
	Ejercicio15Guion();
	Ejercicio16Guion();
	Ejercicio2Guion();

	Ejercicio1Trabajo0();
	Ejercicio2Trabajo0();
	Ejercicio3Trabajo0();
	Ejercicio4Trabajo0();

	return 0;
}