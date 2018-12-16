/*************************************************************************
        > Copyright(c)  NEWPLAN, all rights reserved.
        > File Name   : client.c
        > Author      : NEWPLAN
        > Mail        : newplan001@163.com
        > Created Time: 2018年11月06日 星期二 16时22分01秒
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
static void userHelp(char *str)
{
	printf("%s [server_ip] [server_port]", str);
}
int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		userHelp(argv[0]);
		return 1;
	}

	int sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock < 0)
	{
		perror("sock()");
		exit(-1);
	}
	struct sockaddr_in client;
	client.sin_family = AF_INET;
	client.sin_port = htons(atoi(argv[2]));
	client.sin_addr.s_addr = inet_addr(argv[1]);

	if (connect(sock, (struct sockaddr *)&client, sizeof(client)) < 0)
	{
		perror("connect()");
		exit(-2);
	}
	size_t index = 0;
	struct timeval start, stop, diff;
	gettimeofday(&start, 0);
	while (1)
	{
		char buffer[1024] = "hello world!";
		ssize_t s = 1024;
		//printf("please input:");
		//fflush(stdout);
		//ssize_t s = read(0, buffer, sizeof(buffer) - 1);
		if (s > 0)
		{
			buffer[s - 1] = 0;
			write(sock, buffer, strlen(buffer));
			ssize_t _s = read(sock, buffer, sizeof(buffer) - 1);
			if (_s > 0 && 0)
			{
				buffer[_s] = 0;
				printf("server echo# %s\n", buffer);
			}
		}
		if (index++ % 1000000 == 0)
		{
			gettimeofday(&stop, 0);
			long res = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
			printf("index: %lu, rate: %10f MB/s\n", index, 1.024 * 1.024 * 8 * 1000000.0 / 1024 / (res / 1000000.0));
			start = stop;
		}
	}
	return 0;
}
