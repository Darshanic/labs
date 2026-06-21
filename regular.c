
SEARCHING AND SORTING 

Insertion Sort - Part 1 
 
#include <stdio.h> 
void print(int ar_size, int* ar) { 
int i; 
for(i=0; i<ar_size; i++) { 
printf("%d ", ar[i]); 
} 
printf("\n"); 
} 
#include <string.h> 
#include <math.h> 
#include <stdlib.h> 
#include <assert.h> 
/* Head ends here */ 
void insertionSort(int ar_size, int *  ar) { 
int j = ar_size-1; 
int v = ar[j]; 
while(v < ar[j-1]) { 
ar[j] = ar[j-1]; 
j--; 
print(ar_size, ar); 
} 
ar[j] = v; 
print(ar_size, ar); 
} 
/* Tail starts here */ 
int main() { 
int _ar_size; 
scanf("%d", &_ar_size); 
int _ar[_ar_size], _ar_i; 
for(_ar_i = 0; _ar_i < _ar_size; _ar_i++) {  
scanf("%d", &_ar[_ar_i]);  
} 
insertionSort(_ar_size, _ar); 
return 0; 
} 
-------------------------------------------------------------------

Insertion Sort - Part 2 

 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <stdlib.h> 
#include <assert.h> 
/* Head ends here */ 
void insertionSort(int ar_size, int *  ar) { 
for (int i = 1; i < ar_size; ++i) { 
int j = i - 1; 
int p = ar[i]; 
while (j >= 0 && p < ar[j]) { 
ar[j+1] = ar[j]; 
j--; 
} 
ar[j+1] = p; 
printf("%d", ar[0]); 
for (int k = 1; k < ar_size; ++k) { 
printf(" %d", ar[k]); 
} 
printf("\n"); 
} 
} 
/* Tail starts here */ 
int main() { 
int _ar_size; 
scanf("%d", &_ar_size); 
int _ar[_ar_size], _ar_i; 
for(_ar_i = 0; _ar_i < _ar_size; _ar_i++) {  
scanf("%d", &_ar[_ar_i]);  
} 
insertionSort(_ar_size, _ar); 
return 0; 
} 


-----------------------------------------------------------------

 1. Compare the triplets: 

#include <math.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <assert.h> 
#include <limits.h> 
#include <stdbool.h> 
 
int main(){ 
    int a0;  
    int a1;  
    int a2;  
    scanf("%d %d %d",&a0,&a1,&a2); 
    int b0;  
    int b1;  
    int b2; 
   int a_score = 0; 
    int b_score = 0; 
    scanf("%d %d %d",&b0,&b1,&b2); 
    if (a0 > b0) 
        a_score++; 
    else if (a0 < b0) 
        b_score++; 
    else{} 
        //no op 
    if (a1 > b1) 
        a_score++; 
    else if (a1 < b1) 
        b_score++; 
    else {} 
//no op 
if (a2 > b2) 
a_score++; 
else if (a2 < b2) 
b_score++; 
else {} 
//no op 
printf("%d %d",a_score, b_score);             
return 0; 
} 

==================================================================
2. Diagonal Difference

#include <math.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <assert.h> 
#include <limits.h> 
#include <stdbool.h> 
int main() 
{ 
int n, j; 
int i=0,RightDiagonalSum=0,LeftDiagonalSum=0, firstarray, secondarray; 
scanf("%d",&n); 
int a[n][n]; 
for( firstarray = 0; firstarray < n; firstarray++) 
{ 
for( secondarray = 0; secondarray < n; secondarray++) 
{ 
scanf("%d",&a[firstarray][secondarray]); 
} 
} 
while(i<n) 
{ 
RightDiagonalSum=RightDiagonalSum+a[i][i]; 
i++; 
} 
j=n-1,i=0; 
while(i<n) 
{ 
LeftDiagonalSum=LeftDiagonalSum+a[i][j]; 
i++; 
j--; 
} 
printf("%d",abs(RightDiagonalSum-LeftDiagonalSum)); 
return 0; 
} 
==========================================================



Counting Sort 1 
 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <stdlib.h> 
int main() {  
int n,i; 
int b[100],a; 
scanf("%d",&n); 
for(i=0;i<100;i++) 
{    
b[i]=0; 
} 
for(i=0;i<n;i++) 
{     
scanf("%d",&a); 
b[a]++; 
} 
for(i=0;i<100;i++) 
{    
printf("%d ", b[i]); 
} 
return 0; 
} 
------------------------------------------------------



 3. Non-Divisible Subset 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
// Helper functions for min and max 
int min(int a, int b) { 
return a < b ? a : b; 
} 
int max(int a, int b) { 
return a > b ? a : b; 
} 
int main() { 
int n, k, a, total = 0; 
// Read input n and k 
scanf("%d %d", &n, &k); 
// Array to store counts of each remainder from 0 to k-1 
int mods[k]; 
for (int i = 0; i < k; i++) { 
mods[i] = 0; 
} 
// Read all numbers and count their remainders 
for (int i = 0; i < n; i++) { 
scanf("%d", &a); 
mods[a % k]++; 
} 
// Add at most one element with remainder 0 
total += min(1, mods[0]); 
// Handle complementary remainders 
for (int d = 1; d < (k + 1) / 2; d++) { 
// Pick the maximum count between remainder d and remainder k-d 
total += max(mods[d], mods[k - d]); 
} 
// If k is even, add at most one element with remainder k/2 
if (k % 2 == 0) { 
total += min(1, mods[k / 2]); 
} 
// Print the size of the largest non-divisible subset 
printf("%d\n", total); 
return 0; 
} 

=================================================
 counter game


 #include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <stdlib.h> 
int isPow2(long unsigned  int); 
unsigned long int largePow(long unsigned int); 
int main() { 
int t,i,win; 
long unsigned int n; 
scanf("%d",&t); 
for(i=0;i<t;++i) 
{ 
win=0; 
scanf("%lu",&n); 
if(n==1) 
printf("Richard\n"); 
else 
{ 
while(n!=1) 
{ 
if(isPow2(n)) 
n>>=1; 
else 
n-=largePow(n); 
++win; 
            } 
        } 
        if(win%2==0) 
            printf("Richard\n"); 
        else 
            printf("Louise\n"); 
    } 
    return 0; 
} 
int isPow2(long unsigned int n) 
    { 
    return !(n&(n-1)); 
} 
long unsigned int largePow(long unsigned int n) 
    { 
    long unsigned int m; 
    while(n) 
        { 
        m=n; 
        n=n&(n-1); 
    } 
    return m; 
} 
 
 
 
 
====================================
 
Sherlock and Cost 
 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <stdbool.h> 
 
int main() { 
    int T,N,B,L,R,ML,MR,X,Y,P,Q; 
    scanf("%d",&T); 
    for(int i = 0; i < T; i++) { 
        scanf("%d",&N); 
        for(int j = 0; j < N; j++) { 
            scanf("%d",&B); 
            if(j) { 
                X = L - 1 + ML; 
                Y = R - 1 + MR; 
                P = abs(L - B) + ML; 
                Q = abs(R - B) + MR; 
                ML = (X > Y ? X : Y); 
                MR = (P > Q ? P : Q); 
            } else { 
                ML = MR = 0; 
            } 
            L = 1; 
            R = B; 
        } 
        printf("%d\n", (ML > MR ? ML : MR)); 
    } 
    return 0; 
} 
 
========================================== 
Marc's Cakewalk 
 
#include <math.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <assert.h> 
#include <limits.h> 
#include <stdbool.h> 
void swap(int *a,int *b) 
{ 
    int temp; 
    temp = *a; 
    *a = *b; 
    *b=temp; 
} 
int partition(int *x,int start,int end) 
{ 
    int pivot,pindex,i; 
    pivot = x[end]; 
    pindex = start; 
    for(i=start;i<end;i++) 
    { 
        if(x[i]>=pivot) 
          { 
            swap(&x[i],&x[pindex]); 
            pindex = pindex + 1; 
          } 
} 
swap(&x[pindex],&x[end]); 
return pindex; 
} 
void quicksort(int *x,int start,int end) 
{ 
if(start<end) 
{ 
int i = partition(x,start,end); 
quicksort(x,start,i-1); 
quicksort(x,i+1,end); 
} 
} 
int main(){ 
int n, calories_i, *calories;  
int i; 
int sum = 0; 
scanf("%d",&n); 
calories = malloc(sizeof(int) * n); 
for(calories_i = 0; calories_i < n; calories_i++) 
{ 
} 
scanf("%d",&calories[calories_i]); 
quicksort(calories,0,n-1); 
for(i=0;i<n;i++) 
{ 
sum += calories[i]*((int)pow(2,i)); 
}     
printf("%d",sum); 
// your code goes here 
return 0; 
} 

======================================
  Bear and Steady Gene 


#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <stdlib.h> 
// Function to check if all characters outside the window are balanced (<= n/4) 
int is_balanced(int count[], int target) { 
if (count['A'] > target || count['C'] > target || count['G'] > target || count['T'] > target) { 
return 0; // Not balanced 
} 
return 1; // Balanced 
} 
int steadyGene(char* gene, int n) { 
int target = n / 4; 
int count[128] = {0}; // Using ASCII values as indices for counts 
// Initial pass: Count all characters 
for (int i = 0; i < n; i++) { 
count[gene[i]]++; 
} 
// If already steady, the required substring length is 0 
if (is_balanced(count, target)) { 
return 0; 
} 
int min_len = n; 
int left = 0; 
// Sliding window: right pointer expands the window 
for (int right = 0; right < n; right++) { 
// Decrease count of the character entering the window (remaining outside the balanced 
region) 
count[gene[right]]--; 
// Inner while loop: left pointer shrinks the window from the left 
// as long as the remaining characters outside the window are balanced 
while (is_balanced(count, target)) { 
// A valid window is found, update the minimum length 
// The length of the substring to be replaced is (right - left + 1) 
if ((right - left + 1) < min_len) { 
min_len = (right - left + 1); 
} 
// Shrink the window from the left: increase count of the character leaving the window 
count[gene[left]]++; 
left++; 
} 
} 
return min_len; 
} 
int main() { 
int n; 
// Read the gene length 
if (scanf("%d", &n) != 1) return 1; 
char* gene = (char*)malloc((n + 1) * sizeof(char)); 
// Read the gene string 
if (scanf("%s", gene) != 1) { 
free(gene); 
return 1; 
} 
int result = steadyGene(gene, n); 
printf("%d\n", result); 
free(gene); 
return 0; 
} 

=======================================
 
Pangrams 
 
#include <stdio.h> 
char s[10000]; 
int main() 
{ 
gets(s); 
int f[300]={0},ans=0,i; 
int l=strlen(s); 
for(i=0;i<l;i++) 
{ 
if(s[i]==' ') 
continue; 
if(s[i]>='a') 
s[i]=s[i]-('a'-'A'); 
if(!(f[s[i]]++)) 
ans++; 
} 
if(ans!=26) 
printf("not "); 
printf("pangram\n"); 
return 0; 
} 


 =================================================

  Caesar Cipher 

#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <stdlib.h> 
int main() { 
int n,i,j,k; 
char ar[101]; 
unsigned char x; 
scanf("%d",&n); 
scanf("%s",ar); 
scanf("%d",&k); 
for(i=0;i<n;i++) 
{ 
x=ar[i]; 
if(x>=97 && x<=122) 
{ 
x=x+(k%26); 
            if(x>122) 
            { 
                x=96+(x-122); 
            } 
            ar[i]=x; 
        } 
        else if(x>=65 && x<=90) 
        { 
            x=x+(k%26); 
            if(x>90) 
            { 
                x=64+(x-90); 
            } 
            ar[i]=x; 
        } 
    } 
    printf("%s",ar); 
    return 0; 
} 
 
 
 
 
 

 


========================================
RANGE QUERIES 
 

Prefix Sum Array 


Populate a Prefix-Sum Array from an array of integer elements. 
Also create a function to calculate the sum of elements between a given 
range using the Prefix-Sum Array.  


 Sample Input
5
1 2 3 4 5
1 3
Output
Original array    1    2    3    4    5
Prefix sum array  1    3    6    10   15
The sum is 9
 

 
#include<stdio.h>

void display(int arr[], int n) {

    int i;

    for (i = 0; i < n; i++) {
        printf("\t%d", arr[i]);
    }
}

void create_prefix_sum_array(int arr[], int n) {

    int i;

    for (i = 1; i < n; i++) {
        arr[i] = arr[i] + arr[i - 1];
    }
}

void sum(int arr[], int n, int a, int b) {

    int result;

    if (a == 0)
        result = arr[b];
    else
        result = arr[b] - arr[a - 1];

    printf("\nThe sum is %d", result);
}

int main() {

    int n, i, arr[10], a, b;

    printf("Enter the number of elements\n");
    scanf("%d", &n);

    printf("Enter the array elements\n");

    for (i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    printf("\nOriginal array");
    display(arr, n);

    create_prefix_sum_array(arr, n);

    printf("\nPrefix sum array");
    display(arr, n);

    printf("\nEnter the range to find the sum: ");
    scanf("%d%d", &a, &b);

    sum(arr, n, a, b);

    return 0;
}
=======================================

Fenwick Tree Construction 

Populate a Fenwick Tree from an array of integer elements. 
Given the value of a node in a Fenwick Tree 
tree[k] = sumq (k - p(k) + 1, k) 
Where, p(k) = k&-k  and denotes the largest power of two that divides k  

Sample Input
5
1 2 3 4 5
Output
Original array:
1 2 3 4 5

Fenwick Tree:
1 3 3 10 5


#include<stdio.h>

int sum(int arr[], int a, int b) {

    int i, s = 0;

    for(i = a; i <= b; i++) {
        s += arr[i];
    }

    return s;
}

void create_fenwick_tree(int T[], int arr[], int n) {

    int a, b, k;

    for(k = 1; k <= n; k++) {

        a = k - (k & -k) + 1;
        b = k;

        T[k] = sum(arr, a, b);
    }
}

void display(int arr[], int n) {

    int i;

    for(i = 1; i <= n; i++) {
        printf("%d ", arr[i]);
    }
}

int main() {

    int n, i, arr[100], T[100];

    printf("Enter the number of elements\n");
    scanf("%d", &n);

    printf("Enter the array elements\n");

    for(i = 1; i <= n; i++) {
        scanf("%d", &arr[i]);
    }

    printf("\nOriginal array:\n");
    display(arr, n);

    create_fenwick_tree(T, arr, n);

    printf("\nFenwick Tree:\n");
    display(T, n);

    return 0;
}
-----------------------------------

sum of element

Sample Input
5
Output
Fenwick Tree:
1 4 4 16 6 7 4 29

The sum is 22
 

#include<stdio.h>

void sum(int T[], int k) {

    int s = 0;

    while(k >= 1) {

        s += T[k];

        k -= (k & -k);
    }

    printf("\nThe sum is %d", s);
}

void display(int arr[], int n) {

    int i;

    for(i = 1; i <= n; i++) {
        printf("%d ", arr[i]);
    }
}

int main() {

    int n = 8, k;

    int T[100] = {0, 1, 4, 4, 16, 6, 7, 4, 29};

    printf("\nFenwick Tree:\n");

    display(T, n);

    printf("\nEnter the value for k: ");

    scanf("%d", &k);

    sum(T, k);

    return 0;
}
