#include "serial.h"
#include <signal.h>
#include <iostream>
#include <thread>
#include <mutex>
using namespace std;

bool stop;
mutex stop_mtx;

void listener(int fd)
{
    char buffer[100];
    while(true)
    {
        if(stop_mtx.try_lock())
        {
            if(stop)
            {
                stop_mtx.unlock();
                return;
            }
            stop_mtx.unlock();
        }
        if(read(fd, buffer, 1) > 0)
        {
            cout << "Data read = " << (int)buffer[0] << ' ';
        }
        else
        {
            switch(errno)
            {
                case EAGAIN:
                    //cout << "Blocked" << endl;
                    break;
                default:
                    cout << "Error = " << errno << endl;
            }
        }
        errno = 0;
    }
}

void signal_handler(int sig)
{
    if(sig == SIGINT)
    {
        stop_mtx.lock();
        stop = true;
        stop_mtx.unlock();
    }
}

void submit(int fd)
{
    const char *str = "Hello, world!\n";
    commit(fd, str, sizeof(str));
    cout << "Data submitted" << endl;
}

int main()
{
    stop = false;
    signal(SIGINT, signal_handler);
    int fd = 0;
    InitSerial(fd);
    thread Tlistener(listener, fd);
    sleep(1);
    submit(fd);
    Tlistener.join();
    return 0;
}
