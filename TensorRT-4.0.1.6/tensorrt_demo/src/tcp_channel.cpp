#include <tcp_channel.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

TCPChannel::TCPChannel(std::string nm)
{
    this->name_ = nm;
    LOG(INFO) << " Create TCPChannel " << name_;
}

TCPChannel::~TCPChannel()
{
    LOG(INFO) << name_ << " TCP release";
}

std::string TCPChannel::display()
{
    LOG(INFO) << "This data collector brand is: " << name_;
    return name_;
}

void TCPChannel::client_handler(int new_fd, BlockingQueue<int> *to[2])
{
    LOG(INFO) << "process request from: " << new_fd;
    while (1)
    {
        to[1]->pop("parsing too slow, IO exceeds the overall performance ..."); //receive from cycle path...
        char buffer[1024];
        int recv_c = read(new_fd, buffer, sizeof(buffer));
        //int recv_c = send(new_fd, buffer, sizeof(buffer), 0);
        LOG_IF(FATAL, recv_c == -1) << "receive error...";
        /*
        LOG(INFO) << "received from client " << new_fd << ": " << buffer;
        */
        //std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (write(new_fd, buffer, sizeof(buffer)) != sizeof(buffer))
        {
            LOG(FATAL) << "send error...";
        }
        to[0]->push(new_fd);
        //send(new_fd, buffer, sizeof(buffer), 0);
    }
    close(new_fd);

    //detach()
}

void TCPChannel::setup(BlockingQueue<int> *from[2], BlockingQueue<int> *to[2])
{
    LOG(INFO) << "setup TCP channel ... " << std::endl;

    this->ip_addr = "0.0.0.0";
    this->port = 12345;
    {
        listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in local;
        local.sin_family = AF_INET;
        local.sin_port = htons(port);
        local.sin_addr.s_addr = inet_addr(ip_addr.c_str());
        int sock_opt = 1;
        if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (char *)&sock_opt, sizeof(sock_opt)) < 0)
        {
            LOG(FATAL) << "Set sock port reuse failed ...";
        }
        if (bind(listen_fd, (struct sockaddr *)&local, sizeof(local)) < 0)
        {
            LOG(FATAL) << "error in bind sock to IP and port...";
        }
        if (listen(listen_fd, 10) == -1)
        {
            LOG(FATAL) << "Listen error ...";
        }
    }
    {
        for (int index = 0; index < 1000; index++)
        {
            to[1]->push(index);
        }
    }
    {
        while (1)
        {
            struct sockaddr_in client;
            socklen_t len = sizeof(client);
            int new_fd = accept(listen_fd, (struct sockaddr *)&client, &len);
            CHECK((new_fd) >= 0) << "error in accept new fd";
            printf("Get client from: %s, %d\n", inet_ntoa(client.sin_addr), ntohs(client.sin_port));

            std::thread *tmp = new std::thread(&TCPChannel::client_handler, this, new_fd, to);
            _handler.push_back(tmp);
            tmp->detach();
        }
    }
}