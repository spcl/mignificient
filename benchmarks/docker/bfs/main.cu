#include <chrono>
#include <stdio.h>

#define MAX_THREADS_PER_BLOCK 512

#include "bfs.cu"

int no_of_nodes;
int edge_list_size;
FILE *fp;
 

void ReadGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
    auto s = std::chrono::high_resolution_clock::now();
    no_of_nodes=0;
    edge_list_size=0;
    ReadGraph( argc, argv);

    auto e = std::chrono::high_resolution_clock::now();
	auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000000.0;
    printf("time: %.8f\n", d);
}

void Usage(int argc, char**argv)
{
    fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);
}

void ReadGraph( int argc, char** argv)
{
    char *input_f;
    if(argc != 2)
    {
        Usage(argc, argv);
        exit(0);
    }
    
    input_f = argv[1];
    printf("Reading File\n");
    fp = fopen(input_f, "r");
    if (!fp)
    {
        printf("Error reading graph file\n");
        return;
    }

    int source = 0;

    // Store to no_of_nodes
    fscanf(fp, "%d", &no_of_nodes);

    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;

    if(no_of_nodes>MAX_THREADS_PER_BLOCK)
    {
        num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK;  
    }

	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

    int start, edgeno;
    for(unsigned int i = 0; i < no_of_nodes; i++)
    {
        fscanf(fp, "%d %d", &start, &edgeno);
        h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
    }

    //read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

    //set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    

	printf("Read File\n");

    // MOVED to here
    int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;

    // Call bfs.cu and pass ptr and size
    BFSGraph(h_graph_nodes,
            h_graph_edges,
            h_graph_mask,
            h_graph_visited,
            h_updating_graph_mask,
            h_cost,
            no_of_nodes,
            edge_list_size,
            num_of_blocks,
            num_of_threads_per_block);
}