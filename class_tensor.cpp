#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <ap_fixed.h>


typedef float data_t; 
typedef ap_fixed<16, 8> data_t; 

struct Shape {
    int N; // Batch size
    int H; // Height
    int W; // Width
    int C; // Channels
};


template <typename T>
class TensorMem {
private:
    T* m_data;       
    Shape m_shape;  
    int m_size;      // Tổng số phần tử

public:
    // Constructor
    TensorMem(T* data, Shape shape) : m_data(data), m_shape(shape) {
        #pragma HLS INLINE
        m_size = shape.N * shape.H * shape.W * shape.C;
    }


    int get_index(int n, int h, int w, int c) {
        #pragma HLS INLINE
        return n * (m_shape.H * m_shape.W * m_shape.C) + 
               h * (m_shape.W * m_shape.C) + 
               w * (m_shape.C) + 
               c;
    }

    int read(int n, int h, int w, int c) {
        #pragma HLS INLINE
        int idx = get_index(n, h, w, c);
        return m_data[idx];
    }


    void write(int n, int h, int w, int c, T val) {
        #pragma HLS INLINE
        int idx = get_index(n, h, w, c);
        m_data[idx] = val;
    }


    void load_tile_to_stream(int n, int h_start, int w_start, int h_size, int w_size, hls::stream<T>& out_stream) {
        for (int h = 0; h < h_size; ++h) {
            for (int w = 0; w < w_size; ++w) {
                for (int c = 0; c < m_shape.C; ++c) {
                    #pragma HLS PIPELINE II=1
                    int idx = get_index(n, h_start + h, w_start + w, c);
                    T val = m_data[idx];
                    out_stream.write(val);
                }
            }
        }
    }


    void store_stream_to_mem(int n, int h_start, int w_start, int h_size, int w_size, hls::stream<T>& in_stream) {
        for (int h = 0; h < h_size; ++h) {
            for (int w = 0; w < w_size; ++w) {
                for (int c = 0; c < m_shape.C; ++c) {
                    #pragma HLS PIPELINE II=1
                    int idx = get_index(n, h_start + h, w_start + w, c);
                    if (!in_stream.empty()) {
                        T val = in_stream.read();
                        m_data[idx] = val;
                    }
                }
            }
        }
    }
};


