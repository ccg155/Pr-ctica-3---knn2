// Tarea 0 (1/2): vector dinamico con push_back (A) frente a vector con tamano fijo (B).
// Misma logica que el codigo del enunciado. Cambios minimos: SIZE se puede fijar con -DSIZE=...
// y se imprime una suma de comprobacion para que el compilador (-O2) no elimine los bucles.
// Compilar: g++ -O0|-O2 -DSIZE=1000000 tarea0_vector.cpp -o tarea0_vector
#include <iostream>
#include <vector>
#include <ctime>

#ifndef SIZE
#define SIZE 1000000
#endif

using namespace std;

int main()
{
    // A:
    clock_t begin = clock();
    // declaramos un vector sin inicializar.
    std::vector<int> a;
    // usamos push_back para anadir cada elemento.
    for(int i=0;i<SIZE;i++)
    {
        a.push_back(i);
    }
    double aTime = double(clock() - begin) / CLOCKS_PER_SEC;
    cout<<"Elapsed time: "<<aTime<<"s"<<endl;

    // B:
    begin = clock();
    // declaramos un vector inicializado al tamano requerido.
    std::vector<int> b(SIZE);
    for(int i=0;i<SIZE;i++)
    {
        b[i] = i;
    }
    double bTime = double(clock() - begin) / CLOCKS_PER_SEC;
    cout<<"Elapsed time: "<<bTime<<"s"<<endl;
    cout<<"Gain: "<<aTime/bTime<<endl;
    cout<<"Check: "<<(a[SIZE-1] + b[SIZE-1])<<endl;
}
