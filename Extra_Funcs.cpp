#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "Model.h"


using namespace std;

void split(vector<string>& to_push, string to_split)
{
    int start_index = 0;
    int stop_index = 0;
    while (to_split.length() != start_index + 1)
    {
        if (to_split[start_index] == ',')
        {
            stop_index = start_index + 1;
            while (to_split[stop_index] != ',' && stop_index != to_split.length() - 1)
            {
                stop_index++;
            }

            string temp;
            for (int i = start_index + 1; i < stop_index; i++)
            {
                temp += (to_split[i]);
            }

            if (stop_index == to_split.length() - 1) temp += (to_split[stop_index]);

            start_index = stop_index;
            to_push.push_back(temp);
        }
    }
}

void split_row(vector<double>& to_push, string to_split)
{
    int start_index = 0;
    int stop_index = 0;
    int actual_start = 0;
    while (to_split[actual_start] != ',')
    {
        actual_start++;
    }
    actual_start++;

    start_index = actual_start, stop_index = actual_start;

    while (to_split.length() != start_index + 1)
    {
        if (to_split[start_index] == ',' || start_index == actual_start)
        {
            stop_index = start_index + 1;

            while (to_split[stop_index] != ',' && stop_index != to_split.length() - 1)
            {
                stop_index++;
            }


            string temp;
            if (start_index == actual_start) temp += to_split[actual_start];
            for (int i = start_index + 1; i < stop_index; i++)
            {
                temp += (to_split[i]);
            }

            if (stop_index == to_split.length() - 1) temp += (to_split[stop_index]);

            double num_to_push = stod(temp);
            //cout << num_to_push << endl;


            start_index = stop_index;
            to_push.push_back(num_to_push);
        }
    }
}
