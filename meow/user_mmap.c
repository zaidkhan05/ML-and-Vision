#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#define PAGE_SIZE 4096

int main(int argc, char *argv[]) {
    int fd;
    unsigned char *p_map;

    // Open the /proc file
    fd = open("/proc/mydir/myinfo", O_RDWR);
    if (fd < 0) {
        perror("Failed to open /proc/mydir/myinfo");
        exit(1);
    } else {
        printf("Opened /proc/mydir/myinfo successfully\n");
    }

    // Map the kernel memory to user space
    p_map = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p_map == MAP_FAILED) {
        perror("Failed to mmap");
        close(fd);
        exit(1);
    }

    // Read initial data from the mapped memory
    printf("Initial data in kernel buffer: ");
    for (int i = 0; i < 12; i++) {
        printf("%d ", p_map[i]);
    }
    printf("\n");

    // Modify the mapped memory
    strcpy((char *)p_map, "Hello Kernel");
    printf("Data written to kernel buffer: %s\n", p_map);

    // Cleanup
    if (munmap(p_map, PAGE_SIZE) == -1) {
        perror("Failed to munmap");
    }
    close(fd);

    return 0;
}
