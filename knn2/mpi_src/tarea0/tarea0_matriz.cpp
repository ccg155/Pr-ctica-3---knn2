// Tarea 0 (2/2): recorrido de una matriz por columnas (A) frente a por filas (B).
// Misma logica que el codigo del enunciado. Cambios minimos: SIZE se fija con -DSIZE=...; las
// matrices son estaticas (en el enunciado estan en la pila y con SIZE grande la desbordan) y se
// imprime una suma de comprobacion para que el compilador (-O2) no elimine los bucles.
// Compilar: g++ -O0|-O2 -DSIZE=2000 tarea0_matriz.cpp -o tarea0_matriz
#include <iostream>
#include <vector>
#include <ctime>

#ifndef SIZE
#define SIZE 1000
#endif

using namespace std;

static int a[SIZE][SIZE];
static int b[SIZE][SIZE];

int main()
{
    // A: una matriz en C se almacena como un array de una unica dimension por filas.
    // Si la recorremos por columnas visitamos las celdas en el orden 1 4 7 2 5 8 3 6 9 ...
    clock_t begin = clock();
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
            a[j][i] = i+j;
    }
    double aTime = double(clock() - begin) / CLOCKS_PER_SEC;
    cout<<"Elapsed time: "<<aTime<<"s"<<endl;

    // B: por filas visitamos las celdas en el orden 1 2 3 4 5 6 7 8 9 ...
    begin = clock();
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
            b[i][j] = i+j;
    }
    double bTime = double(clock() - begin) / CLOCKS_PER_SEC;
    cout<<"Elapsed time: "<<bTime<<"s"<<endl;
    cout<<"Gain: "<<aTime/bTime<<endl;
    cout<<"Check: "<<(a[SIZE-1][SIZE-1] + b[SIZE-1][SIZE-1])<<endl;
}
