#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "mpi.h"

/*
Preform 2D neighbor MPI send/recv

ranks layed out in following order:
____________
 0 | 1 | 2 |
____________
 3 | 4 | 5 |
____________
 6 | 7 | 8 |
____________
*/

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Assume layout is square
    int lin_dim = sqrt(size);
    if(lin_dim*lin_dim != size) {
      printf("MPI size must be perfect square");
        return(EXIT_FAILURE);
    }

    // Define neighbors
    // upper left/middle/right
    // mid left, mid right
    // lower left/middle/right
    int ul,um,ur,ml,mr,ll,lm,lr;
    int x,y,n_x,n_y,neighbor_rank;

    // Current rank's xy coord
    x = rank % lin_dim;
    y = floor(rank/lin_dim);

    // Find neighbors
    n_x = x-1;
    n_y = y-1;
    neighbor_rank = n_y*lin_dim + n_x;
    ul = (n_x>=0 && n_y>=0)?neighbor_rank:MPI_PROC_NULL;
    
    n_x = x;
    n_y = y-1;
    neighbor_rank = n_y*lin_dim + n_x;
    um = n_y>=0?neighbor_rank:MPI_PROC_NULL;

    n_x = x+1;
    n_y = y-1;
    neighbor_rank = n_y*lin_dim + n_x;
    ur = (n_y>=0 && n_x<lin_dim)?neighbor_rank:MPI_PROC_NULL;   

    n_x = x-1;
    n_y = y;
    neighbor_rank = n_y*lin_dim + n_x;
    ml = n_x>=0?neighbor_rank:MPI_PROC_NULL;   

    n_x = x+1;
    n_y = y;
    neighbor_rank = n_y*lin_dim + n_x;
    mr = n_x<lin_dim?neighbor_rank:MPI_PROC_NULL;

    n_x = x-1;
    n_y = y+1;
    neighbor_rank = n_y*lin_dim + n_x;
    ll = (n_x>=0 && n_y<lin_dim)?neighbor_rank:MPI_PROC_NULL;

    n_x = x;
    n_y = y+1;
    neighbor_rank = n_y*lin_dim + n_x;
    lm = n_y<lin_dim?neighbor_rank:MPI_PROC_NULL; 

    n_x = x+1;
    n_y = y+1;
    neighbor_rank = n_y*lin_dim + n_x;
    lr = (n_x<lin_dim && n_y<lin_dim)?neighbor_rank:MPI_PROC_NULL;

    int neighbors[8];
    neighbors[0] = ul;
    neighbors[1] = um;
    neighbors[2] = ur;
    neighbors[3] = ml;
    neighbors[4] = mr;
    neighbors[5] = ll;
    neighbors[6] = lm;
    neighbors[7] = lr;

    // Allocate send/recv buffers
    // Send buffers contain the sending rank
    int *send_buffs[8];
    int *recv_buffs[8];
    int count = 1000;
    size_t size_buff = count*sizeof(int);
    int i, j;
    for(i=0; i<8; i++) {
        send_buffs[i] = malloc(size_buff);
        recv_buffs[i] = malloc(size_buff);
        for(j=0; j<count; j++) {
            send_buffs[i][j] = rank;
            recv_buffs[i][j] = -1;
        }
    }

    MPI_Request send_requests[8];
    MPI_Request recv_requests[8];


    MPI_Barrier(MPI_COMM_WORLD);

    double start, end;
    start = MPI_Wtime();

    int k,l;
    for(k=0; k<100; k++) {
        int err;
        // Post Recvs from all neighbors
        for(i=0; i<8; i++) {
            err = MPI_Irecv(recv_buffs[i], count, MPI_INT, neighbors[i], j, MPI_COMM_WORLD, &recv_requests[i]);
            if(err != MPI_SUCCESS) {
              printf("ERROR: MPI_IRecv error %d", err);
            }
        }

        // Initiate Send to all  neighbors
        for(i=0; i<8; i++) {
            err = MPI_Isend(send_buffs[i], count, MPI_INT, neighbors[i], j, MPI_COMM_WORLD, &send_requests[i]);
            if(err != MPI_SUCCESS) {
              printf("ERROR: MPI_ISend error %d", err);
            }
        }

        // Wait to complete 
        err = MPI_Waitall(8, recv_requests, MPI_STATUSES_IGNORE);   
        if(err != MPI_SUCCESS) {
          printf("ERROR: MPI_Waitall error %d", err);
        }

        // Test that received values are sum of neighboring ranks multiplied by the count
        int sum=0;
        for(i=0; i<8; i++) {
          if(neighbors[i] != MPI_PROC_NULL) {
            for(j=0; j<count; j++) {
              if(recv_buffs[i][j] != neighbors[i]) {
                printf("ERROR: expected %d but received %d\n", neighbors[i], recv_buffs[i][j]);
                return(EXIT_FAILURE);
              }
           }
         }
       }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if(rank==0)
        printf("Total comm time: %f seconds\n", end-start);

    MPI_Finalize();
    return 0;
}
