/*
 MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


 C IMPLEMENTATION

 ********************  Conversions between color systems *******************************

 This module defines bidirectional conversions of color values between colors
 expressed in the RGB (Red Green Blue) color space used in computer monitors and three
 HLS (Hue Lightness Saturation).



*/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <assert.h>
#include <time.h>


struct hsl{
    double h;    // hue
    double s;    // saturation
    double l;    // Lightness
};

struct rgb{
    double r;   // red
    double g;   // green
    double b;   // blue
};

struct rgba{
    double r;   // red
    double g;   // green
    double b;   // blue
    double a;   // alpha
};


// METHOD 1 (USING POINTER)
// Convert RGB color model into HSL and reciprocally
double * rgb_to_hsl(double r, double g, double b);
double * hsl_to_rgb(double h, double s, double l);
double hue_to_rgb(double m1, double m2, double h);

// METHOD 2 (USING STRUCT)
struct hsl struct_rgb_to_hsl(double r, double g, double b);
struct rgb struct_hsl_to_rgb(double h, double s, double l);

// DETERMINE MAX & MIN VALUES FROM A PIXEL DEFINE WITH RGB VALUES
double fmax_rgb_value(double red, double green, double blue);
double fmin_rgb_value(double red, double green, double blue);

// CONVENIENT CONSTANTS
#define ONE_SIX     1.0 / 6.0
#define ONE_THIRD   1.0 / 3.0
#define TWO_THIRD   2.0 / 3.0
#define ONE_255     1.0 / 255.0
#define ONE_360     1.0 / 360.0

#define max_c(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

// All inputs have to be double precision (python float) in range [0.0 ... 255.0]
// Output: return the maximum value from given RGB values (double precision).
inline double fmax_rgb_value(double red, double green, double blue)
{
    if (red>green){
        if (red>blue) {
		    return red;
	}
		else {
		    return blue;
	    }
    }
    else if (green>blue){
	    return green;
	}
    else {
	    return blue;
	}
}

// All inputs have to be double precision (python float) in range [0.0 ... 255.0]
// Output: return the minimum value from given RGB values (double precision).
inline double fmin_rgb_value(double red, double green, double blue)
{
    if (red<green){
        if (red<blue){
            return red;
        }
    else{
	    return blue;}
    }
    else if (green<blue){
	    return green;
	}
    else{
	    return blue;
	}
}


/*
******************************* POINTER METHOD *******************************
*/

// HSL: (Hue - position in the spectrum, Saturation, Lightness)
// All inputs have to be double precision, (python float) in range [0.0 ... 1.0]
// Outputs is a C array pointer containing HSL values (double precision) normalized.
// If you require the HSL value in the format (Hue in degrees, Saturation in percentage,
// Lightness in percentage)  multiply HSL values such as
// h = h * 360 (°)
// s = s * 100 (%)
// l = l * 100 (%)
inline double * rgb_to_hsl(double r, double g, double b)
{ 
// check if all inputs are normalized
//    assert ((0.0<= r) <= 1.0);
//    assert ((0.0<= g) <= 1.0);
//    assert ((0.0<= b) <= 1.0);

    double *hsl = malloc (sizeof (double)* 3);
    // Check if the memory has been successfully 
    // allocated by malloc or not 
    if (hsl == NULL) {
        printf("Memory not allocated.\n"); 
        exit(0); 
    } 
    double cmax=0, cmin=0, delta=0, t;
    cmax = fmax_rgb_value(r, g, b);
    cmin = fmin_rgb_value(r, g, b);
    delta = (cmax - cmin);
   
    double h, l, s;
    l = (cmax + cmin) / 2.0;  
    
    if (delta == 0) {
    h = 0;
    s = 0;   
    }
    else {
    	  if (cmax == r){
    	        t = (g - b) / delta;
    	        if ((fabs(t) > 6.0) && (t > 0.0)) {
                  t = fmod(t, 6.0);
                }
                else if (t < 0.0){
                t = 6.0 - fabs(t);
                }

	            h = 60.0 * t;
          }
    	  else if (cmax == g){
                h = 60.0 * (((b - r) / delta) + 2.0);
          }
          
    	  else if (cmax == b){
    	        h = 60.0 * (((r - g) / delta) + 4.0);
          }
	  
    	  if (l <=0.5) {
	            s=(delta/(cmax + cmin));
	      }
	  else {
	        s=(delta/(2.0 - cmax - cmin));
	  }     
    }
    
    hsl[0] = h * ONE_360;
    hsl[1] = s;
    hsl[2] = l;
    return hsl;
}


inline double hue_to_rgb(double m1, double m2, double h)
{
    if ((fabs(h) > 1.0) && (h > 0.0)) {
      h = fmod(h, 1.0);
    }
    else if (h < 0.0){
    h = 1.0 - fabs(h);
    }

    if (h < ONE_SIX){
        return m1 + (m2 - m1) * h * 6.0;
    }
    if (h < 0.5){
        return m2;
    }
    if (h < TWO_THIRD){
        return m1 + ( m2 - m1 ) * (TWO_THIRD - h) * 6.0;
    }
    return m1;
}


// Convert HSL color model into RGB (red, green, blue)
// all inputs have to be double precision, (python float) in range [0.0 ... 1.0]
// outputs is a C array pointer containing RGB values (double precision) normalized.
// If you need RGB values within range [0 ... 255] multiply the values by 255
inline double * hsl_to_rgb(double h, double s, double l)
{
    double *rgb = malloc (sizeof (double) * 3);
    // Check if the memory has been successfully 
    // allocated by malloc or not 
    if (rgb == NULL) { 
        printf("Memory not allocated.\n"); 
        exit(0); 
    } 

    double m2=0, m1=0;

    if (s == 0.0){
        rgb[0] = l;
        rgb[1] = l;
        rgb[2] = l;
        return rgb;
    }
    if (l <= 0.5){
        m2 = l * (1.0 + s);
    }
    else{
        m2 = l + s - (l * s);
    }
    m1 = 2.0 * l - m2;
    
    rgb[0] = hue_to_rgb(m1, m2, (h + ONE_THIRD));
    rgb[1] = hue_to_rgb(m1, m2, h);
    rgb[2] = hue_to_rgb(m1, m2, (h - ONE_THIRD));
    return rgb;
}

/*
******************************* END OF POINTER METHOD *******************************
*/


/*
******************************* STRUCT METHOD *******************************
*/

// HSL: (Hue - position in the spectrum, Saturation, Lightness)
// All inputs have to be double precision, (python float) in range [0.0 ... 1.0]
// Outputs is a C struct containing HSL values (double precision) normalized.
// If you require the HSL value in the format (Hue in degrees, Saturation in percentage,
// Lightness in percentage)  multiply HSL values such as
// h = h * 360 (°)
// s = s * 100 (%)
// l = l * 100 (%)
inline struct hsl struct_rgb_to_hsl(double r, double g, double b)
{
// check if all inputs are normalized
//    assert ((0.0 <= r) || (r <= 1.0));
//    assert ((0.0 <= g) || (r <= 1.0));
//    assert ((0.0 <= b) || (r <= 1.0));

    struct hsl hsl_={.h=0.0, .s=0.0, .l=0.0};

    double cmax=0, cmin=0, delta=0, t;
    cmax = fmax_rgb_value(r, g, b);
    cmin = fmin_rgb_value(r, g, b);
    delta = (cmax - cmin);

    double h, l, s;
    l = (cmax + cmin) / 2.0;

    if (delta == 0) {
    h = 0;
    s = 0;
    }
    else {
    	  if (cmax == r){
    	        t = (g - b) / delta;
    	        if ((fabs(t) > 6.0) && (t > 0.0)) {
                  t = fmod(t, 6.0);
                }
                else if (t < 0.0){
                t = 6.0 - fabs(t);
                }

	            h = 60.0 * t;
          }
    	  else if (cmax == g){
                h = 60.0 * (((b - r) / delta) + 2.0);
          }

    	  else if (cmax == b){
    	        h = 60.0 * (((r - g) / delta) + 4.0);
          }

    	  if (l <=0.5) {
	            s=(delta/(cmax + cmin));
	      }
	  else {
	        s=(delta/(2.0 - cmax - cmin));
	  }
    }

    hsl_.h = h * ONE_360;
    hsl_.s = s;
    hsl_.l = l;
    return hsl_;
}


// Convert HSL color model into RGB (red, green, blue)
// All inputs have to be double precision, (python float) in range [0.0 ... 1.0]
// Outputs is a C struct containing RGB values (double precision) normalized.
// If you need RGB values within range [0 ... 255] multiply the values by 255
inline struct rgb struct_hsl_to_rgb(double h, double s, double l)
{

    struct rgb rgb_={.r=0.0, .g=0.0, .b=0.0};

    double m2=0, m1=0;

    if (s == 0.0){
        rgb_.r = l;
        rgb_.g = l;
        rgb_.b = l;
        return rgb_;
    }
    if (l <= 0.5){
        m2 = l * (1.0 + s);
    }
    else{
        m2 = l + s - (l * s);
    }
    m1 = 2.0 * l - m2;

    rgb_.r = hue_to_rgb(m1, m2, (h + ONE_THIRD));
    rgb_.g = hue_to_rgb(m1, m2, h);
    rgb_.b = hue_to_rgb(m1, m2, (h - ONE_THIRD));
    return rgb_;
}
/*
******************************* END OF STRUCT METHOD *******************************
*/

int main(){
double *array;
double *arr;
double h, l, s; 
double r, g, b;
int i = 0, j = 0, k = 0;

int n = 1000000;
double *ptr;
clock_t begin = clock();

/* here, do your time-consuming job */
for (i=0; i<=n; ++i){
ptr = rgb_to_hsl(25.0/255.0, 60.0/255.0, 128.0/255.0);
}
clock_t end = clock();
double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
printf("\ntotal time %f :", time_spent); 

printf("\nTesting algorithm(s).");
n = 0;

for (i=0; i<256; i++){
    for (j=0; j<256; j++){
        for (k=0; k<256; k++){

            array = rgb_to_hsl(i/255.0, j/255.0, k/255.0);
            h = array[0];
            s = array[1];
            l = array[2];
            free(array);
            arr = hsl_to_rgb(h, s, l);
            r = round(arr[0] * 255.0);
            g = round(arr[1] * 255.0);
            b = round(arr[2] * 255.0);
	        free(arr);
            // printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
            // printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
            // printf("\n %f, %f, %f ", h, l, s);
	    
            if (abs(i - r) > 0.1) {   
                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
                    printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
                printf("\n %f, %f, %f ", h, l, s);
                        n+=1;
                return -1;
            }
            if (abs(j - g) > 0.1){    
                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
                    printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
                printf("\n %f, %f, %f ", h, l, s);
                        n+=1;
                return -1;
            }

            if (abs(k - b) > 0.1){
                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
                printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
                printf("\n %f, %f, %f ", h, l, s);
                n+=1;
		        return -1;
                               
            }
 	   
            }

        }
    }


printf("\nError(s) found n=%i", n);
return 0;
}

