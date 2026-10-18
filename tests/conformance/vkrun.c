// vkrun: execute one SPIR-V compute entry point on the first Vulkan device the loader exposes.
//
// usage: vkrun <module.spv> <entry> <gx> <gy> <gz> <binding>:<in|out|inout>:<file>[:<bytes>] ...
//   All buffers are storage buffers in descriptor set 0. For `in`/`inout` the file is uploaded;
//   for `out`/`inout` the buffer is written back to the same path (an `out` buffer needs :<bytes>).
//   The driver is selected by the Vulkan loader (VK_DRIVER_FILES / VK_ICD_FILENAMES).
//   On success prints "ok gpu_ns=<n>" (timestamp-query delta, 0 if unsupported) followed by one
//   "key=value" line each for driver, driver_info, api and device; on failure prints "error <msg>".
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x)                                                         \
  do {                                                                   \
    VkResult r_ = (x);                                                   \
    if (r_ != VK_SUCCESS) {                                              \
      printf("error %s failed (%d)\n", #x, (int)r_);                     \
      exit(2);                                                           \
    }                                                                    \
  } while (0)

typedef struct {
  uint32_t binding;
  int mode;  // 1 = in, 2 = out, 3 = inout
  char path[1024];
  VkDeviceSize size;
  VkBuffer buf;
  VkDeviceMemory mem;
} Buf;

static void *read_file(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *d = malloc(len > 0 ? (size_t)len : 1);
  *n = fread(d, 1, (size_t)len, f);
  fclose(f);
  return d;
}

static uint32_t find_mem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  printf("error no host-visible memory type\n");
  exit(2);
}

int main(int argc, char **argv) {
  if (argc < 7) {
    printf("error usage\n");
    return 1;
  }
  const char *spv_path = argv[1], *entry = argv[2];
  uint32_t gx = (uint32_t)atoi(argv[3]), gy = (uint32_t)atoi(argv[4]), gz = (uint32_t)atoi(argv[5]);
  int nb = argc - 6;
  Buf *bufs = calloc((size_t)nb, sizeof(Buf));
  for (int i = 0; i < nb; i++) {
    char tmp[1200];
    strncpy(tmp, argv[6 + i], sizeof tmp - 1);
    char *b = strtok(tmp, ":"), *m = strtok(NULL, ":"), *p = strtok(NULL, ":"), *s = strtok(NULL, ":");
    if (!b || !m || !p) {
      printf("error bad buffer spec %s\n", argv[6 + i]);
      return 1;
    }
    bufs[i].binding = (uint32_t)atoi(b);
    bufs[i].mode = !strcmp(m, "in") ? 1 : !strcmp(m, "out") ? 2 : 3;
    strncpy(bufs[i].path, p, sizeof bufs[i].path - 1);
    bufs[i].size = s ? (VkDeviceSize)atoll(s) : 0;
  }

  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "vkrun", 1, "none", 1, VK_API_VERSION_1_3};
  // Portability enumeration is required for MoltenVK and absent on native drivers (e.g. Mesa lavapipe).
  uint32_t n_iext = 0;
  vkEnumerateInstanceExtensionProperties(NULL, &n_iext, NULL);
  VkExtensionProperties *iprops = calloc(n_iext ? n_iext : 1, sizeof *iprops);
  vkEnumerateInstanceExtensionProperties(NULL, &n_iext, iprops);
  int has_portability = 0;
  for (uint32_t i = 0; i < n_iext; i++)
    if (!strcmp(iprops[i].extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) has_portability = 1;
  const char *iexts[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
                         VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.flags = has_portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = has_portability ? 2 : 0;
  ici.ppEnabledExtensionNames = iexts;
  VkInstance inst;
  CHECK(vkCreateInstance(&ici, NULL, &inst));
  uint32_t npd = 1;
  VkPhysicalDevice pd;
  vkEnumeratePhysicalDevices(inst, &npd, &pd);
  if (npd == 0) {
    printf("error no device\n");
    return 2;
  }
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pd, &props);
  uint32_t nq = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
  VkQueueFamilyProperties qf[16];
  if (nq > 16) nq = 16;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
  uint32_t qfi = 0;
  for (uint32_t i = 0; i < nq; i++)
    if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      qfi = i;
      break;
    }
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfi, 1, &prio};
  uint32_t n_dext = 0;
  vkEnumerateDeviceExtensionProperties(pd, NULL, &n_dext, NULL);
  VkExtensionProperties *dprops = calloc(n_dext ? n_dext : 1, sizeof *dprops);
  vkEnumerateDeviceExtensionProperties(pd, NULL, &n_dext, dprops);
  int has_subset = 0;
  for (uint32_t i = 0; i < n_dext; i++)
    if (!strcmp(dprops[i].extensionName, "VK_KHR_portability_subset")) has_subset = 1;
  const char *dexts[] = {"VK_KHR_portability_subset"};
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = has_subset ? 1 : 0;
  dci.ppEnabledExtensionNames = dexts;
  VkDevice dev;
  CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, qfi, 0, &q);

  size_t spv_n = 0;
  uint32_t *spv = read_file(spv_path, &spv_n);
  if (!spv) {
    printf("error cannot read %s\n", spv_path);
    return 1;
  }
  VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, spv_n, spv};
  VkShaderModule sm;
  CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));

  VkDescriptorSetLayoutBinding *lb = calloc((size_t)nb, sizeof *lb);
  for (int i = 0; i < nb; i++) {
    lb[i].binding = bufs[i].binding;
    lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    lb[i].descriptorCount = 1;
    lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, (uint32_t)nb, lb};
  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
  VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 0, NULL};
  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
  VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = sm;
  cpci.stage.pName = entry;
  cpci.layout = pl;
  VkPipeline pipe;
  CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

  for (int i = 0; i < nb; i++) {
    size_t n = 0;
    void *data = NULL;
    if (bufs[i].mode & 1) {
      data = read_file(bufs[i].path, &n);
      if (!data) {
        printf("error cannot read %s\n", bufs[i].path);
        return 1;
      }
      if (!bufs[i].size) bufs[i].size = n;
    }
    if (!bufs[i].size) {
      printf("error buffer %u has no size\n", bufs[i].binding);
      return 1;
    }
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, bufs[i].size,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    CHECK(vkCreateBuffer(dev, &bci, NULL, &bufs[i].buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, bufs[i].buf, &mr);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size,
                                find_mem(pd, mr.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    CHECK(vkAllocateMemory(dev, &mai, NULL, &bufs[i].mem));
    CHECK(vkBindBufferMemory(dev, bufs[i].buf, bufs[i].mem, 0));
    void *mapped;
    CHECK(vkMapMemory(dev, bufs[i].mem, 0, bufs[i].size, 0, &mapped));
    memset(mapped, 0, (size_t)bufs[i].size);
    if (data) memcpy(mapped, data, n < bufs[i].size ? n : (size_t)bufs[i].size);
    vkUnmapMemory(dev, bufs[i].mem);
    free(data);
  }

  VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nb};
  VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
  VkDescriptorPool dp;
  CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
  VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl};
  VkDescriptorSet ds;
  CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  VkDescriptorBufferInfo *dbi = calloc((size_t)nb, sizeof *dbi);
  VkWriteDescriptorSet *w = calloc((size_t)nb, sizeof *w);
  for (int i = 0; i < nb; i++) {
    dbi[i].buffer = bufs[i].buf;
    dbi[i].range = VK_WHOLE_SIZE;
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = ds;
    w[i].dstBinding = bufs[i].binding;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(dev, (uint32_t)nb, w, 0, NULL);

  VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2, 0};
  VkQueryPool qp;
  CHECK(vkCreateQueryPool(dev, &qpci, NULL, &qp));
  VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, qfi};
  VkCommandPool cp;
  CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
  VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp,
                                      VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
  VkCommandBuffer cb;
  CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
  VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                   VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
  CHECK(vkBeginCommandBuffer(cb, &cbbi));
  vkCmdResetQueryPool(cb, qp, 0, 2);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
  vkCmdDispatch(cb, gx, gy, gz);
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
  CHECK(vkEndCommandBuffer(cb));
  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
  CHECK(vkQueueWaitIdle(q));
  uint64_t ts[2] = {0, 0};
  double gpu_ns = 0;
  if (vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, sizeof(uint64_t),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
    gpu_ns = (double)(ts[1] - ts[0]) * props.limits.timestampPeriod;

  for (int i = 0; i < nb; i++) {
    if (!(bufs[i].mode & 2)) continue;
    void *mapped;
    CHECK(vkMapMemory(dev, bufs[i].mem, 0, bufs[i].size, 0, &mapped));
    FILE *f = fopen(bufs[i].path, "wb");
    if (!f) {
      printf("error cannot write %s\n", bufs[i].path);
      return 1;
    }
    fwrite(mapped, 1, (size_t)bufs[i].size, f);
    fclose(f);
    vkUnmapMemory(dev, bufs[i].mem);
  }
  VkPhysicalDeviceDriverProperties driver = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  if (props.apiVersion >= VK_API_VERSION_1_2) {
    VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
    vkGetPhysicalDeviceProperties2(pd, &props2);
  }
  printf("ok gpu_ns=%.0f\n", gpu_ns);
  printf("driver=%s\n", driver.driverName[0] ? driver.driverName : "unknown");
  printf("driver_info=%s\n", driver.driverInfo);
  printf("api=%u.%u.%u\n", VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
         VK_API_VERSION_PATCH(props.apiVersion));
  printf("device=%s\n", props.deviceName);
  return 0;
}
