#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <bits/stdc++.h>
#include <vector>
#include <cfloat>
#include <cstring>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

struct ThreadData {
    int id;
    int num_threads;
    int k;
    ArffData* train;
    ArffData* test;
    int* predictions;
};

// Calculates the distance between two instances
float distance(float* instance_A, float* instance_B, int num_attributes) {
    float sum = 0;
    
    for (int i = 0; i < num_attributes-1; i++) {
        float diff = instance_A[i] - instance_B[i];
        sum += diff*diff;
    }
    
    return sqrt(sum);
}

// The function that each thread will execute
void* knn_thread_func(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    // Get dataset properties
    int num_classes = data->train->num_classes();
    int num_attributes = data->train->num_attributes();
    int train_num_instances = data->train->num_instances();
    int test_num_instances = data->test->num_instances();

    float* train_matrix = data->train->get_dataset_matrix();
    float* test_matrix = data->test->get_dataset_matrix();

    // Determine the chunk of test instances this thread will process
    int chunk_size = test_num_instances / data->num_threads;
    int start_index = data->id * chunk_size;
    int end_index = (data->id == data->num_threads - 1) ? test_num_instances : start_index + chunk_size;

    // These arrays are local to each thread, so no race conditions
    float* candidates = (float*) calloc(data->k * 2, sizeof(float));
    int* classCounts = (int*)calloc(num_classes, sizeof(int));

    // Process the assigned chunk of test instances
    for (int queryIndex = start_index; queryIndex < end_index; queryIndex++) {
        for(int i = 0; i < 2 * data->k; i++){ candidates[i] = FLT_MAX; }
        memset(classCounts, 0, num_classes * sizeof(int));

        for (int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
            float dist = distance(&test_matrix[queryIndex * num_attributes], &train_matrix[keyIndex * num_attributes], num_attributes);

            for (int c = 0; c < data->k; c++) {
                if (dist < candidates[2 * c]) {
                    for (int x = data->k - 2; x >= c; x--) {
                        candidates[2 * x + 2] = candidates[2 * x];
                        candidates[2 * x + 3] = candidates[2 * x + 1];
                    }
                    candidates[2 * c] = dist;
                    candidates[2 * c + 1] = train_matrix[keyIndex * num_attributes + num_attributes - 1];
                    break;
                }
            }
        }

        for (int i = 0; i < data->k; i++) {
            classCounts[(int)candidates[2 * i + 1]]++;
        }

        int max_value = -1;
        int max_class = 0;
        for (int i = 0; i < num_classes; i++) {
            if (classCounts[i] > max_value) {
                max_value = classCounts[i];
                max_class = i;
            }
        }
        // Write prediction to the shared array. No lock needed as each thread has a unique index range.
        data->predictions[queryIndex] = max_class;
    }

    free(candidates);
    free(classCounts);
    pthread_exit(NULL);
}

// Implements a threaded kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int* KNN(ArffData* train, ArffData* test, int k, int num_threads) {    

    int* predictions = (int*)calloc(test->num_instances(), sizeof(int));
    
    /*************************************************************
    *** Complete this code and return the array of predictions ***
    **************************************************************/
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].k = k;
        thread_data[i].train = train;
        thread_data[i].test = test;
        thread_data[i].predictions = predictions;
        pthread_create(&threads[i], NULL, knn_thread_func, (void*)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    return predictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) { // for each instance compare the true class and predicted class    
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++) {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return 100 * successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 5)
    {
        printf("Usage: ./program datasets/train.arff datasets/test.arff k num_threads");
        exit(0);
    }

    // k value for the k-nearest neighbors
    int k = strtol(argv[3], NULL, 10);
    int num_threads = strtol(argv[4], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();
    
    struct timespec start, end;
    int* predictions = NULL;
    
    // Initialize time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    predictions = KNN(train, test, k, num_threads);
    
    // Stop time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    uint64_t time_difference = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The %i-NN classifier for %lu test instances and %lu train instances required %llu ms CPU time for threaded with %d threads. Accuracy was %.2f\%\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) time_difference, accuracy, num_threads);

    free(predictions);
    free(confusionMatrix);
}

/*  // Example to print the test dataset
    float* test_matrix = test->get_dataset_matrix();
    for(int i = 0; i < test->num_instances(); i++) {
        for(int j = 0; j < test->num_attributes(); j++)
            printf("%.0f, ", test_matrix[i*test->num_attributes() + j]);
        printf("\n");
    }
*/