#include <stdio.h>
#include <stdlib.h>
#include<xmmintrin.h>
#include "mkl.h"
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <string.h>
#include<math.h>
#include<iostream>
#include <papi.h>
#define PAPI_ERROR_CHECK(X) if((X)!=PAPI_OK) std::cerr<<X<<" Error \n";
using namespace std;
double time_in_mill_now();
//icpc -Wno-write-strings -g -std=c++0x -O3 -qopenmp -vec-report1 -restrict -pthread  -Denable_gpu -Denable_mkl -m64 -w -I"/home/singh.980/intel/mkl/include"  -O3 -parallel -qopenmp -Ofast -xCORE-AVX2  lower_k.cc -c
double time_in_mill_now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double time_in_mill =
    (tv.tv_sec) * 1000.0 + (tv.tv_usec) / 1000.0;
  return time_in_mill;
}
#define MFACTOR (32)
#define LOG_MFACTOR (5)
#define MFACTOR16 (16)
#define LOG_MFACTOR16 (4)
#define MFACTOR8 (8)
#define LOG_MFACTOR8 (3)
#define CORE 4
#define TYPE double
#define BC (64)
#define BR (64)
#define CACHE 35*1024*128//65536
#define PAD 16
//#define THRESHOLD (6)
int SM_K = 8;
int SM_WIDTH = 64;
bool HIGH_SD = false;
// int THRESHOLD = -1;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))

struct v_struct {
    int row, col;
    int grp;
    short idx;
    TYPE val;
};

#define BF (4)
#define BSIZE (BF*32)


//#define SM_WIDTH (384)
#define DBSIZE (256*4)

int numThread=CORE;

int iterMax=2048;
int cache = CACHE;


void cacheFlush(double *X, double *Y) {

  for(int i=0; i<20*1000000; i++) {
    X[i]=Y[i]+rand() % 5;
    Y[i] += rand() % 7;

  }

}


int pcnt[100000];
int nr0, nr, nc, ne, np, nn, upper_size;
int gold_ne;
struct v_struct *temp_v;
struct v_struct *gold_temp_v;
struct v_struct *new_temp_v;
int *pb;
//int *dc, *dc;
int dcnt;

int nseg;
int *dc, *didx;
int *d;
TYPE *dv;

int sc;

int compare1(const void *a, const void *b)
{
    if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp > 0) return 1;
    if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp < 0) return -1;
    if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
    if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
    return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}
//temp_v[i].grp = temp_v[i].row / SM_WIDTH;
int compare2(const void *a, const void *b)
{
    if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp > 0) return 1;
    if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp < 0) return -1;
    if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col > 0) return 1;
    if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col < 0) return -1;
    return ((struct v_struct *)a)->row - ((struct v_struct *)b)->row;
}


void ready(int argc, char **argv)
{
    FILE *fp;
    int *loc;
    char buf[300];
    int nflag, sflag;
    int dummy, pre_count=0, tmp_ne;
    int i;
    
    srand(time(NULL));
    
    //sc = atoi(argv[2]);
    fp = fopen(argv[1], "r");
    fgets(buf, 300, fp);
    if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
    else sflag = 0;
    if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
    else if(strstr(buf, "complex") != NULL) nflag = -1;
    else nflag = 1;
    
#ifdef SYM
    sflag = 1;
#endif
    
    while(1) {
        pre_count++;
        fgets(buf, 300, fp);
        if(strstr(buf, "%") == NULL) break;
    }
    fclose(fp);
    
    fp = fopen(argv[1], "r");
    for(i=0;i<pre_count;i++)
        fgets(buf, 300, fp);
    
    fscanf(fp, "%d %d %d", &nr, &nc, &ne);
    nr0 = nr;
    ne *= (sflag+1);
    nr = CEIL(nr, BF)*BF;
    nc = CEIL(nc, BF)*BF;
    cout<<"Original Matrix nr="<<nr<<" nc="<<nc<<" ne"<<ne<<endl;
    
    temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));
    gold_temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));
    
    for(i=0;i<ne;i++) {
        fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
        temp_v[i].grp = 0;
        temp_v[i].row--; temp_v[i].col--;
        
        if(temp_v[i].row < 0 || temp_v[i].row >= nr || temp_v[i].col < 0 || temp_v[i].col >= nc) {
            fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
            exit(0);
        }
        if(nflag == 0) temp_v[i].val = (TYPE)(rand()%1048576)/1048576;
        else if(nflag == 1) {
            TYPE ftemp;
            fscanf(fp, " %lf ", &ftemp);
            temp_v[i].val = ftemp;
        } else { // complex
            TYPE ftemp1, ftemp2;
            fscanf(fp, " %lf %lf ", &ftemp1, &ftemp2);
            temp_v[i].val = ftemp1;
        }
        if(sflag == 1) {
            i++;
            temp_v[i].row = temp_v[i-1].col;
            temp_v[i].col = temp_v[i-1].row;
            temp_v[i].val = temp_v[i-1].val;
            temp_v[i].grp = 0;
        }
    }
    qsort(temp_v, ne, sizeof(struct v_struct), compare1);
    
    loc = (int *)malloc(sizeof(int)*(ne+1));
    
    memset(loc, 0, sizeof(int)*(ne+1));
    loc[0]=1;
    for(i=1;i<ne;i++) {
        if(temp_v[i].row == temp_v[i-1].row && temp_v[i].col == temp_v[i-1].col)
            loc[i] = 0;
        else loc[i] = 1;
    }
    for(i=1;i<=ne;i++)
        loc[i] += loc[i-1];
    for(i=ne; i>=1; i--)
        loc[i] = loc[i-1];
    loc[0] = 0;
    
    for(i=0;i<ne;i++) {
        temp_v[loc[i]].row = temp_v[i].row;
        temp_v[loc[i]].col = temp_v[i].col;
        temp_v[loc[i]].val = temp_v[i].val;
        temp_v[loc[i]].grp = temp_v[i].grp;
    }
    ne = loc[ne];
    temp_v[ne].row = nr;
    gold_ne = ne;
    for(i=0;i<=ne;i++) {
        gold_temp_v[i].row = temp_v[i].row;
        gold_temp_v[i].col = temp_v[i].col;
	//////////////////////////////////////////
        gold_temp_v[i].val = (TYPE) 1.0;//temp_v[i].row + temp_v[i].col;//temp_v[i].val;//Change this part later plz just for debugging
	temp_v[i].val = (TYPE) 1.0;//temp_v[i].row + temp_v[i].col;
	/////////////////////////////////////////////
        gold_temp_v[i].grp = temp_v[i].grp;
    }
    free(loc);

    // for(int i =0; i< 100 ; i++)
    // {
    //     cout<<" gold_temp_v[i].val "<<gold_temp_v[i].val<<" temp_v[i].val "<<temp_v[i].val<<endl;
    // }
}


void gen_structure(int Nk, int Tk, int CacheSize)
{
    int i, j;

    int Tj = CacheSize/Tk;
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    int m = nr;
    int k = Nk;

    double mean = 0;
    double sd = 0;
    //    if(SM_WIDTH == 256)
    //    cout<<"Average ele in rows : "<<ne/nc<<endl;


    mean = ne / Nk;
    //    if(SM_WIDTH == 256)
    //    cout<<"\n Mean : "<<mean<<endl;
    int temp_colx = temp_v[0].col;
    double sumOfSquare = 0;
    int rows_in_column = 0;
    int count = 0;

    for (int i = 0; i < ne; i++)
    {

        if(temp_colx != temp_v[i].col)
        {
            double square_dist = (count ) - mean ;

            if (square_dist < 0)
            {
                square_dist = square_dist * (-1);
            }

            sumOfSquare += pow(square_dist, 2);
            count = 1;
            temp_colx = temp_v[i].col;
        }
        else
        {
            count++;
        }

    }

    sd = sumOfSquare / nc;
    sd = sqrt(sd);
    //    if(SM_WIDTH == 256)
    //    cout<<" \nstandard deviation = "<< sd<<endl;

    /*    if(sd > 200)
        {
            SM_WIDTH = 16;
            HIGH_SD = true;
        }
    */
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    nseg = CEIL(nc, Tj);//change
    // cout<<"nseg "<<nseg<<endl;
    pb = (int *)_mm_malloc(sizeof(int) * (nseg + 1), 64);
    memset(pb, -1, sizeof(int) * (nseg + 1));
    //    dc = (int *)malloc(sizeof(int)*ne); // loose UB
    //    didx = (int *)malloc(sizeof(int)*ne); // loose UB

#pragma vector aligned
    #pragma omp parallel for num_threads(CORE)

    for(i = 0; i < ne; i++)
    {
        temp_v[i].grp = temp_v[i].col / Tj;//change

    }


    int * new_pb = (int *)_mm_malloc(sizeof(int) * (nseg + 1), 64);
    pb = (int *)_mm_malloc(sizeof(int) * (nseg + 1), 64);
    int * new_pb3 = (int *)_mm_malloc(sizeof(int) * (nseg + 1), 64);

#pragma vector aligned
    #pragma omp parallel for num_threads(CORE)

    for(int i = 0; i < nseg + 1; i++ )
    {
        new_pb[i] = 0;
    }

    int temp_panel = -2;
    int temp_row = -2;

    for(int i = 0; i < ne ; i++)
    {
        if(temp_v[i].grp != temp_panel || temp_v[i].row != temp_row)
        {
            new_pb[temp_v[i].grp + 1]++;
            temp_panel = temp_v[i].grp;
            temp_row = temp_v[i].row;
        }
    }

    for(int i = 0; i < nseg; i++ )
    {
        new_pb[i + 1] += new_pb[i];

    }

    pb[0] = 0;

#pragma vector aligned
    #pragma omp parallel for num_threads(CORE)

    for(int i = 0; i < nseg + 1; i++ )
    {
        pb[i] = new_pb[i];
        new_pb3[i] = new_pb[i];

    }

    dcnt = new_pb[nseg];
    // cout<<"\n dcnt x = "<<dcnt<<endl;
    dc = (int *)_mm_malloc(sizeof(int) * (dcnt + 128), 64); // loose UB
    didx = (int *)_mm_malloc(sizeof(int) * (dcnt + 128), 64); // loose UB
    int * new_didx2 = (int *)_mm_malloc(sizeof(int) * (dcnt + 128), 64);

#pragma vector aligned
    #pragma omp parallel for num_threads(CORE)

    for(int i = 0; i < dcnt + 128; i++)
    {
        dc[i] = 0;
        didx[i] = 0;
    }

    temp_panel = -2;
    temp_row = -2;
    didx[0] = -1;
#pragma vector aligned

    for(int i = 0; i < ne ; i++)
    {
        if(temp_v[i].grp != temp_panel || temp_v[i].row != temp_row)
        {
            dc[new_pb[temp_v[i].grp]] = temp_v[i].row;
            didx[new_pb[temp_v[i].grp]]++;
            new_pb[temp_v[i].grp]++;
            temp_panel = temp_v[i].grp;
            temp_row = temp_v[i].row;

        }
        else
        {
            didx[new_pb[temp_v[i].grp]]++;
        }
    }

    for(int i = 0; i < dcnt + 127; i++ )
    {
        didx[i + 1] += didx[i];
    }

    didx[dcnt]++;

#pragma vector aligned
    #pragma omp parallel for num_threads(CORE)

    for(int i = 0; i < dcnt + 128; i++)
    {
        new_didx2[i] = didx[i];

    }

    d = (int *)_mm_malloc(sizeof(int) * (ne + 128), 64); // can be char
    dv = (TYPE *)_mm_malloc(sizeof(TYPE) * (ne + 128), 64); // can be char

    temp_panel = pb[temp_v[0].grp];
    temp_row = temp_v[0].row;

#pragma vector aligned
    #pragma omp parallel for num_threads(CORE)

    for(int i = 0; i < ne; i++)
    {
        d[i] = 0;
        dv[i] = 0.0;
    }

    int range = nseg / CORE;
    range++;

#pragma vector aligned
    #pragma omp parallel num_threads(CORE)
    {
        int threadID = omp_get_thread_num();

        for(int i = 0; i < ne ; i++)
        {
            if( ( temp_v[i].grp >= range * threadID ) &&  ( temp_v[i].grp < range * (threadID + 1) ) )
            {
                int didx_index = new_pb3[temp_v[i].grp] ;
                int col = new_didx2[didx_index];
                d[col] = temp_v[i].col;
                dv[col] = temp_v[i].val;
                new_didx2[didx_index] += 1;

                if(new_didx2[didx_index] == new_didx2[didx_index + 1])
                {
                    new_pb3[temp_v[i].grp] += 1;
                }
            }
        }
    }

    _mm_free(new_pb);
    _mm_free(new_pb3);
    _mm_free(new_didx2);

}


//=======================================================================================================================
//=======================================================================================================================
//=======================================================================================================================
//=======================================================================================================================
//=======================================================================================================================

void process(int Nk, int Tk, int CacheSize)
{
    int i, j;
    TYPE *B, *C;
    TYPE *vout_gold;

    double *X = (double*)malloc(20 * 1000000 * sizeof(double));
    double *Y = (double*)malloc(20 * 1000000 * sizeof(double));

    int Tj = CacheSize/Tk;


    double tot_ms;
    sc = Nk;
    {
        {
            int tile_size = Tk;
            cacheFlush(X, Y);
            B = (TYPE *)_mm_malloc(sizeof(TYPE) * nseg * Tj * (Nk + PAD), 64);
            C = (TYPE *)_mm_malloc(sizeof(TYPE) * nr * Nk, 64);

            for(i = 0; i < nr * (Nk) ; i++)
            {
                C[i] = 0.0f;
            }

            vout_gold = (TYPE *)malloc(sizeof(TYPE) * nr * Nk);

            for(i = 0; i < nseg * Tj * (Nk+PAD); i++)
            {
                B[i] = (TYPE) 1.0;//(rand()%10)/10;
            }

            struct timeval starttime, midtime, endtime, timediff;

            double total_time = 0.0, avg_time = 0.0;

            int K = Nk;

            int *segment = pb;

            int *column_number = d;

            int *row_index = didx;

            int *row_number = dc;

            TYPE *A_value = dv;

            //===========================Dense MM============================================================================================
            //===========================Dense MM============================================================================================
            //===========================Dense MM============================================================================================clea
            /*
            __assume_aligned(row_number, 64);

            __assume_aligned(A_value, 64);

            __assume_aligned(C, 64);

            __assume_aligned(B, 64);

            __assume_aligned(row_index, 64);
            */

            for(int iter = 0; iter < 10; iter++)
            {
                cacheFlush(X, Y);
                gettimeofday(&starttime, NULL);


                for (int k = 0; k < K; k += Tk) //slices
                {
                    for(int col_panel = 0; col_panel < nseg; col_panel++)
                    {
                        if(segment[col_panel] == segment[col_panel + 1])
                        {
                            continue;
                        }

                        #pragma vector aligned
                        #pragma omp parallel for num_threads(CORE)
                        for(int j = segment[col_panel]; j < segment[col_panel + 1] ; j++) //seperate dCSC matrices
                        {

                            int rowNumber = row_number[j];
                            int numOfCols = row_index[j + 1] - row_index[j];

                            if(j!=segment[col_panel + 1])
                            {
                                //for (int kk = k; kk < k + tile_size; kk+=LS_W)
                                //__builtin_prefetch (&C[column_number[j+1]*K+kk], 1, 3);
                                //__builtin_prefetch (&C[column_number[j + 1]*K+tile_size], 1, 2);
                                //__builtin_prefetch (&A_value[column_index[j + 1]], 0, 0);
                            }

                            for(int i = 0 ; i < numOfCols ; i ++) // NNZ per active row/col
                            {
                                //for (int kk = k; kk < k + tile_size; kk+=LS_W)
                                    //__builtin_prefetch (&B[row_number[i+LS_W + column_index[j]] * (K + PAD) +kk ], 0, 1);
                            
                                #pragma ivdep
                                #pragma unroll(8)
                                for (int kk = k; kk < k + Tk; kk++)
                                {
                                    C[ rowNumber * K + kk  ]  += (A_value[i + 0 + row_index[j] ] * B[ column_number[i + 0 + row_index[j]] * (K + PAD) + kk ] );
                                }
                            }

                        }

                    }

                }

                gettimeofday(&endtime, NULL);
                double elapsed = ((endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec) / 1000000.0;

                cout << "K= " << K << " Tj=" << Tj << ", Tk= " << Tk << ", Elapsed= " << elapsed << " sec, GFLOPS= " << (double)2 * (double)gold_ne*(double)K / elapsed / 1000000000 << endl;
            }


            //===========================Sparse MM ============================================================================================

            // //fprintf(stdout, "\n\n total ms %lf, GFLOPS %lf,", tot_ms,(double)gold_ne*2*sc/tot_ms/1000000);
            // printf("ne=%d gold_ne=%d nr=%d nr0=%d\n",ne, gold_ne, nr, nr0);
            if(sc > 0)
            {
                for(i = 0; i < nr * sc; i++)
                {
                    vout_gold[i] = 0.0f;
                }

                for(i = 0; i < gold_ne; i++)
                {
                    for(j = 0; j < sc; j++)
                    {
                        vout_gold[gold_temp_v[i].row * sc + j] += B[(sc + PAD) * gold_temp_v[i].col + j] * gold_temp_v[i].val;
                        // cout<<" C " << vout_gold[gold_temp_v[i].row*sc+j] <<"  "<<B[sc*gold_temp_v[i].col+j] << "   "<<gold_temp_v[i].val<<endl;
                    }
                }

                // for(int y=0;y<nr*sc;y++)
                // {
                //     if(vout_gold[y]!=0)
                //         cout<<" non zeros in vout_dold " << vout_gold[y]<<endl;
                // }
                int num_diff = 0;

                for(i = 0; i < nr * sc; i++)
                {
                    //if(i%nr != i%nr0) continue;
                    TYPE p1 = vout_gold[i];
                    TYPE p2 = C[i];

                    if(p1 < 0)
                    {
                        p1 *= -1;
                    }

                    if(p2 < 0)
                    {
                        p2 *= -1;
                    }

                    TYPE diff;
                    diff = p1 - p2;

                    if(diff < 0)
                    {
                        diff *= -1;
                    }

                    if(diff / MAX(p1, p2) > 0.01)
                    {
                        //                if(num_diff < 20*1) fprintf(stdout, "row %d col %d %lf %lf\n",i/sc, i%sc, C[i], vout_gold[i]);
                        num_diff++;
                    }

                    //      cout<<p1<<" ";
                    //      if((i+1)%sc == 0)
                    //          cout<<endl;
                }

                //    fprintf(stdout, "num_diff : %d\n", num_diff);
                if(num_diff != 0)
                {
                    fprintf(stdout, " DIFF %lf, %d\n", (double)num_diff / (nr * sc) * 100, num_diff);
                }

                //    fprintf(stdout, "ne : %d\n", gold_ne);
                //#endif
                //fprintf(stdout, "\n");
            }




            _mm_free(C);
            _mm_free(B);
            free(vout_gold);
        }
        //cout<<endl;
    }
    /*
    _mm_free(pb);
    _mm_free(dc);
    _mm_free(didx);
    _mm_free(d);
    _mm_free(dv);
    free(X);
    free(Y);
    */
}


void papi()
{
    int i,j;
    // nc / 8 = integer??
    TYPE *B, *C;
    TYPE *vout_gold;
    
    // cout<<"pb[nseg]="<<pb[nseg]<<" ne="<<ne<<endl;
    

    //memory allocation for cache-flushing
    double *X = (double*)malloc(20*1000000*sizeof(double));
    double *Y = (double*)malloc(20*1000000*sizeof(double));


    int event_set = PAPI_NULL;
        long_long values[4];
        int retval =    PAPI_library_init(PAPI_VER_CURRENT);

        if(retval != PAPI_VER_CURRENT && retval > 0){
                cout<<retval << " "<< PAPI_VER_CURRENT<<endl;
        }



    if ((retval = PAPI_create_eventset(&event_set)) != PAPI_OK) {
                fprintf(stderr, "PAPI error %d: %s\n",retval,           PAPI_strerror(retval));
                exit(1);
    }

    PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_L1_DCM));
    //cout<<"here"<<endl;
    PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_L2_DCM));
    PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_LD_INS));
    PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_PRF_DM));
 
    double tot_ms;
    // cout<<endl<<"DENSE SPMM"<<endl;
    sc = SM_K;
    //for(int sc = 8; sc <=2048 ; sc*=2)
    {
    //for(int tile_size = 8; tile_size <=sc; tile_size*=2 )
    {
        int tile_size  = cache/SM_WIDTH;
   //     if(sc >=1024)
   //         tile_size = 512;   
        cacheFlush(X, Y);
        B = (TYPE *)_mm_malloc(sizeof(TYPE)*nseg*SM_WIDTH*(sc+PAD),64);
        C = (TYPE *)_mm_malloc(sizeof(TYPE)*nr*sc,64);
        for(i=0;i<nr*sc;i++) {
            C[i] = 0.0f;
        }
        
        vout_gold = (TYPE *)malloc(sizeof(TYPE)*nr*sc);
        
        for(i=0;i<nseg*SM_WIDTH*sc;i++) {
            B[i] = (TYPE)(rand()%1048576)/1048576;
        }  
            
        struct timeval starttime, midtime, endtime,timediff;
        double total_time=0.0, avg_time=0.0;

        int K=sc;

        int *segment=pb;
        int *column_number=dc;
        int *column_index=didx;
        int *row_number=d;
        TYPE *A_value=dv;
        //===========================Dense MM============================================================================================
        //===========================Dense MM============================================================================================
        //===========================Dense MM============================================================================================clea
         
         __assume_aligned(row_number, 64);
         __assume_aligned(A_value, 64);
         __assume_aligned(C, 64);
         __assume_aligned(B, 64);
         __assume_aligned(column_index, 64);        
        
        
        gettimeofday(&starttime,NULL);
        PAPI_ERROR_CHECK(PAPI_start(event_set));

        if(tile_size == K ){
/*            #pragma ivdep//k
            #pragma vector aligned//k
            #pragma temporal (C)
            #pragma omp parallel for num_threads(CORE) schedule(dynamic, 1) */
            for(int row_panel=0; row_panel<nseg; row_panel++)
            {
                if(segment[row_panel]==segment[row_panel+1]) continue;
                
                //faor (int k = 0; k < K; k+=tile_size) //slices vector aligned
                //                    #pragma temporal (C, A_value)
                //                                        #pragma omp parallel for num_threads(CORE)
                //
                //{
                    #pragma vector aligned
                    #pragma temporal (B)
                    #pragma omp parallel for num_threads(CORE)                   
                    for(int j= segment[row_panel]; j< segment[row_panel+1] ; j++) //seperate dCSC matrices 
                    {
                        int colNumber = column_number[j];
                        int numOfRows = column_index[j+1] - column_index[j];
                        int i = 0;


/*                        for(; i < ((int)numOfRows/8) * 8 ; i+=8) 
                        {
                            #pragma ivdep
                            #pragma unroll(8)
                            #pragma vector nontemporal (A_value)
                            //#pragma prefetch C:_MM_HINT_T1
                            //#pragma prefetch B:_MM_HINT_T1
                            #pragma temporal (C)
                            for (int kk = 0; kk < K; kk++)// i, rows of 
                            {
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 0 + column_index[j] ] * B[ row_number[i + 0 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 1 + column_index[j] ] * B[ row_number[i + 1 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 2 + column_index[j] ] * B[ row_number[i + 2 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 3 + column_index[j] ] * B[ row_number[i + 3 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 4 + column_index[j] ] * B[ row_number[i + 4 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 5 + column_index[j] ] * B[ row_number[i + 5 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 6 + column_index[j] ] * B[ row_number[i + 6 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 7 + column_index[j] ] * B[ row_number[i + 7 + column_index[j]] * K + kk ] );
                            /*    C[ row_number[i + 0 + column_index[j]] * K + kk  ] = C[ row_number[i + 0 + column_index[j]] * K + kk  ] + (A_value[i + 0 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 1 + column_index[j]] * K + kk  ] = C[ row_number[i + 1 + column_index[j]] * K + kk  ] + (A_value[i + 1 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 2 + column_index[j]] * K + kk  ] = C[ row_number[i + 2 + column_index[j]] * K + kk  ] + (A_value[i + 2 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 3 + column_index[j]] * K + kk  ] = C[ row_number[i + 3 + column_index[j]] * K + kk  ] + (A_value[i + 3 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 4 + column_index[j]] * K + kk  ] = C[ row_number[i + 4 + column_index[j]] * K + kk  ] + (A_value[i + 4 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 5 + column_index[j]] * K + kk  ] = C[ row_number[i + 5 + column_index[j]] * K + kk  ] + (A_value[i + 5 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 6 + column_index[j]] * K + kk  ] = C[ row_number[i + 6 + column_index[j]] * K + kk  ] + (A_value[i + 6 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 7 + column_index[j]] * K + kk  ] = C[ row_number[i + 7 + column_index[j]] * K + kk  ] + (A_value[i + 7 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                //C[ rowInd * K + kk + kkk ] = C[ rowInd * K + kk + kkk ] + (tx * B[ col * K + kk + kkk ] );
                            
                            }
                        }
                        if( numOfRows - i >= 4)
                        {
                            
                            #pragma ivdep
                            #pragma vector nontemporal (A_value)
                            #pragma prefetch C:_MM_HINT_T1
                            #pragma unroll(8)
                            #pragma temporal (C)
                            for (int kk = 0; kk < K; kk++)// i, rows of 
                            {
                            	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 0 + column_index[j] ] * B[ row_number[i + 0 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 1 + column_index[j] ] * B[ row_number[i + 1 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 2 + column_index[j] ] * B[ row_number[i + 2 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 3 + column_index[j] ] * B[ row_number[i + 3 + column_index[j]] * K + kk ] );
                            /*
                                C[ row_number[i + 0 + column_index[j]] * K + kk  ] = C[ row_number[i + 0 + column_index[j]] * K + kk  ] + (A_value[i + 0 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 1 + column_index[j]] * K + kk  ] = C[ row_number[i + 1 + column_index[j]] * K + kk  ] + (A_value[i + 1 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 2 + column_index[j]] * K + kk  ] = C[ row_number[i + 2 + column_index[j]] * K + kk  ] + (A_value[i + 2 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 3 + column_index[j]] * K + kk  ] = C[ row_number[i + 3 + column_index[j]] * K + kk  ] + (A_value[i + 3 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                
                            }
                            i+=4;
                        }
*/
                        for( ; i < numOfRows ; i ++) 
                        { 
                            #pragma ivdep//k
                            #pragma vector nontemporal (A_value)//k
                            #pragma prefetch C:_MM_HINT_T1
                            #pragma prefetch B:_MM_HINT_T2
                            #pragma unroll(8)
                            #pragma temporal (B)
                            for (int kk = 0; kk < K; kk++)
                            {
                            	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 0 + column_index[j] ] * B[ row_number[i + 0 + column_index[j]] * (K+PAD) + kk ] );
                                //C[ row_number[i + column_index[j]] * K + kk + 0] += (A_value[i + column_index[j] ] * B[ column_number[j] * K + kk + 0 ] );
                            }
                        }

                        
            
                    }
                    
                //}

            }
         }
        else
        {

            
/*            #pragma ivdep
            #pragma vector aligned
            #pragma temporal (C, A_value)
            #pragma omp parallel for num_threads(CORE) schedule(dynamic, 1) 
*/
            for(int row_panel=0; row_panel<nseg; row_panel++)
            {
                if(segment[row_panel]==segment[row_panel+1]) continue;
                
                for (int k = 0; k < K; k+=tile_size) //slices
                {
                    #pragma vector aligned
                    #pragma temporal (B)
                    #pragma omp parallel for num_threads(CORE)                   
                    for(int j= segment[row_panel]; j< segment[row_panel+1] ; j++) //seperate dCSC matrices 
                    {

                        int colNumber = column_number[j];
                        int numOfRows = column_index[j+1] - column_index[j];
                        int i = 0;


/*                        for(; i < ((int)numOfRows/8) * 8 ; i+=8) 
                        {
                            #pragma ivdep
                            #pragma unroll(8)
                            #pragma vector nontemporal (A_value)
                            #pragma prefetch C:_MM_HINT_T1
                            //#pragma prefetch B:_MM_HINT_T1
                            #pragma temporal (C)
                            for (int kk = 0; kk < K; kk++)// i, rows of 
                            {
                            	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 0 + column_index[j] ] * B[ row_number[i + 0 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 1 + column_index[j] ] * B[ row_number[i + 1 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 2 + column_index[j] ] * B[ row_number[i + 2 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 3 + column_index[j] ] * B[ row_number[i + 3 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 4 + column_index[j] ] * B[ row_number[i + 4 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 5 + column_index[j] ] * B[ row_number[i + 5 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 6 + column_index[j] ] * B[ row_number[i + 6 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 7 + column_index[j] ] * B[ row_number[i + 7 + column_index[j]] * K + kk ] );

                            	/*
                                C[ row_number[i + 0 + column_index[j]] * K + kk  ] = C[ row_number[i + 0 + column_index[j]] * K + kk  ] + (A_value[i + 0 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 1 + column_index[j]] * K + kk  ] = C[ row_number[i + 1 + column_index[j]] * K + kk  ] + (A_value[i + 1 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 2 + column_index[j]] * K + kk  ] = C[ row_number[i + 2 + column_index[j]] * K + kk  ] + (A_value[i + 2 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 3 + column_index[j]] * K + kk  ] = C[ row_number[i + 3 + column_index[j]] * K + kk  ] + (A_value[i + 3 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 4 + column_index[j]] * K + kk  ] = C[ row_number[i + 4 + column_index[j]] * K + kk  ] + (A_value[i + 4 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 5 + column_index[j]] * K + kk  ] = C[ row_number[i + 5 + column_index[j]] * K + kk  ] + (A_value[i + 5 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 6 + column_index[j]] * K + kk  ] = C[ row_number[i + 6 + column_index[j]] * K + kk  ] + (A_value[i + 6 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 7 + column_index[j]] * K + kk  ] = C[ row_number[i + 7 + column_index[j]] * K + kk  ] + (A_value[i + 7 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                //C[ rowInd * K + kk + kkk ] = C[ rowInd * K + kk + kkk ] + (tx * B[ col * K + kk + kkk ] );
                                
                            }
                        }
                        if( numOfRows - i >= 4)
                        {
                            
                            #pragma ivdep
                            #pragma vector nontemporal (A_value)
                            #pragma prefetch C:_MM_HINT_T1
                            #pragma unroll(8)
                            #pragma temporal (C)
                            for (int kk = 0; kk < K; kk++)// i, rows of 
                            {
                            
                            	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 0 + column_index[j] ] * B[ row_number[i + 0 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 1 + column_index[j] ] * B[ row_number[i + 1 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 2 + column_index[j] ] * B[ row_number[i + 2 + column_index[j]] * K + kk ] );
                              	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 3 + column_index[j] ] * B[ row_number[i + 3 + column_index[j]] * K + kk ] );
                              	/*
                                C[ row_number[i + 0 + column_index[j]] * K + kk  ] = C[ row_number[i + 0 + column_index[j]] * K + kk  ] + (A_value[i + 0 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 1 + column_index[j]] * K + kk  ] = C[ row_number[i + 1 + column_index[j]] * K + kk  ] + (A_value[i + 1 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 2 + column_index[j]] * K + kk  ] = C[ row_number[i + 2 + column_index[j]] * K + kk  ] + (A_value[i + 2 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                C[ row_number[i + 3 + column_index[j]] * K + kk  ] = C[ row_number[i + 3 + column_index[j]] * K + kk  ] + (A_value[i + 3 + column_index[j] ] * B[ colNumber * K + kk  ] );
                                
                            }
                            i+=4;
                        }
*/
                        for( ; i < numOfRows ; i ++) 
                        { 
                            #pragma ivdep//k
                            #pragma vector nontemporal (A_value)//k
                            #pragma prefetch C:_MM_HINT_T1
                            #pragma prefetch B:_MM_HINT_T2
                            #pragma unroll(8)
                            #pragma temporal (B)
                            for (int kk = k; kk < k+tile_size; kk++)
                            {
                            	C[ colNumber * K + kk  ]  = C[ colNumber * K + kk  ] + (A_value[i + 0 + column_index[j] ] * B[ row_number[i + 0 + column_index[j]] * (K+PAD) + kk ] );
                                //C[ row_number[i + column_index[j]] * K + kk + 0] += (A_value[i + column_index[j] ] * B[ column_number[j] * K + kk + 0 ] );
                            }
                        }
            
                    }
                    
                }

            }
        }
        
        PAPI_ERROR_CHECK(PAPI_stop(event_set, values));
        std::cout << "L1 DCM ,  " << values[0] << ", " << "L2 DCM, " << values[1]<<", " << "L3 TCM, " << values[2]<<", " << "PRF DM, " << values[3] << "\n";

        gettimeofday(&endtime,NULL); 
        double elapsed = ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
    
        cout<<"PAPI, K= "<< K <<" SM_WIDTH="<<SM_WIDTH<< ", slice_size= "<< tile_size << ", Elapsed= " << elapsed << " sec, GFLOPS= " << (double)2*(double)gold_ne*(double)K/elapsed/1000000000 << endl;

    
    //===========================Sparse MM ============================================================================================

        // //fprintf(stdout, "\n\n total ms %lf, GFLOPS %lf,", tot_ms,(double)gold_ne*2*sc/tot_ms/1000000);
        // printf("ne=%d gold_ne=%d nr=%d nr0=%d\n",ne, gold_ne, nr, nr0);
        if(sc > 1)
        {
            for(i=0;i<nr*sc;i++) {
                vout_gold[i] = 0.0f;
            }
            for(i=0;i<gold_ne;i++) {
                for(j=0;j<sc;j++) {
                    vout_gold[gold_temp_v[i].row*sc+j] += B[(sc+PAD)*gold_temp_v[i].col+j] * gold_temp_v[i].val;
                    // cout<<" C " << vout_gold[gold_temp_v[i].row*sc+j] <<"  "<<B[sc*gold_temp_v[i].col+j] << "   "<<gold_temp_v[i].val<<endl;
                }
            }
            // for(int y=0;y<nr*sc;y++)
            // {
            //     if(vout_gold[y]!=0)
            //         cout<<" non zeros in vout_dold " << vout_gold[y]<<endl;
            // }
            int num_diff=0;
            for(i=0;i<nr*sc;i++) {
                //if(i%nr != i%nr0) continue;
                TYPE p1 = vout_gold[i]; TYPE p2 = C[i];
                if(p1 < 0) p1 *= -1;
                if(p2 < 0) p2 *= -1;
                TYPE diff;
                diff = p1 - p2;
                if(diff < 0) diff *= -1;
                if(diff / MAX(p1,p2) > 0.01) {
                    //if(num_diff < 20*1) fprintf(stdout, "row %d col %d %lf %lf\n",i/sc, i%sc, C[i], vout_gold[i]);
                    num_diff++;
                }
            }
            //    fprintf(stdout, "num_diff : %d\n", num_diff);
            if(num_diff != 0)
            fprintf(stdout, " DIFF %lf, %d\n", (double)num_diff/(nr*sc)*100, num_diff);
            //    fprintf(stdout, "ne : %d\n", gold_ne);
            //#endif
            //fprintf(stdout, "\n");
        }
        
            

       
        _mm_free(C);
        _mm_free(B);
        free(vout_gold);
    } 
    //cout<<endl;
    }
//    if(SM_K==64 || SM_K==128 )
    {
        _mm_free(pb);
        _mm_free(dc);
        _mm_free(didx);
        _mm_free(d);
        _mm_free(dv);
        free(X);
        free(Y);
    }

}

int main(int argc, char **argv)
{
    fprintf(stdout,"TTAAGG,%s,",argv[1]);

    ready(argc, argv);
    
    //k=8 to 64
//    int cache = CACHE;// 65536;
    //SM_WIDTH = 256;
    struct timeval starttime, midtime, endtime,timediff;
    double total_time=0.0, avg_time=0.0;
    gettimeofday(&starttime,NULL);

    //gen_structure();

    gettimeofday(&endtime,NULL);
    double elapsed = ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
    cout<<"Preprocess overhead "<<" Elapsed: "<<elapsed<<endl;
    cout<<"================================STATS================================="<<endl;

	int iter = 1;
/*    for(SM_K = 8 ; SM_K <= 2048 ; SM_K*=2) 
    { 
    	SM_WIDTH = cache/SM_K;
    	gen_structure();      
	for(int i=0;i<iter;i++)
 	        process();
        papi();
    }
*/
    SM_K = 1024;

    int Nk, CacheSize, Tk; 
    for(Nk = 128; Nk<=1024; Nk *= 8){
        for(CacheSize = CACHE ; CacheSize <=CACHE ; CacheSize *= 2){
            for(Tk = Nk ; Tk >= 32 ; Tk/=2){
                gen_structure(Nk,Tk,CacheSize);
                process(Nk,Tk,CacheSize);
            }
        }
    }


    /*
for(SM_K=8;SM_K<=2048;SM_K*=2)
    for(cache = CACHE ; cache <= CACHE ; cache*=2)
    {
	SM_WIDTH = MAX(16,cache/SM_K);
        for(; SM_WIDTH <= MAX(cache/SM_K,cache/8); SM_WIDTH *= 2){
                gen_structure();
//                for(int i=0;i<iter;i++)
//                process();
                papi();
        }
    }
*/
/*    int max_k = 2048;
    int min_k = 8;

    	for(SM_WIDTH =  MAX(16,CACHE/4/max_k); SM_WIDTH <= CACHE*4/8; SM_WIDTH *=2){
		gen_structure();
		for(cache = CACHE/4; cache <= CACHE*4; cache *= 2){
			for(SM_K= ;SM_K<= ;SM_K*=2){
				process();
				papi();
			}
		}
	}
*/

    cout<<"================================STATS================================="<<endl;
    //k= 128
    SM_WIDTH = 16;
    SM_K = 128;
/*
//	for(int tile_size = 16; tile_size <= SM_K ; tile_size *= 2)
	for(;SM_WIDTH  <= 4096 ; SM_WIDTH *= 4){
		for(int tile_size = 16; tile_size <= SM_K ; tile_size *= 2)
    //    int tile_size = SM_K;
        {
		    	gen_structure();
		    	process();
                papi();
		}
	}


 //   knlpapi();
    cout<<"================================STATS================================="<<endl;
    //k= 256 to 2048
    SM_WIDTH = 64;
//    gen_structure();

    for(SM_K = 256 ; SM_K <= 2048 ; SM_K*=2)
    {        
    //    process();
    }
*/
}
