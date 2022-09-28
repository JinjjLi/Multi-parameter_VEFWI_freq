# Set project directory one level above of Makefile directory. $(CURDIR) is a GNU make variable containing the path to the current working directory
PROJDIR := $(realpath $(CURDIR)/..)
SOURCEDIR := $(PROJDIR)/src
BUILDDIR := $(PROJDIR)/obj

# Name of the final executable
TARGET = test_Inv

# Decide whether the commands will be shwon or not
VERBOSE = FALSE

# Create the list of directories
DIRS = Main General Setup Forward Inversion 
SOURCEDIRS = $(foreach dir, $(DIRS), $(addprefix $(SOURCEDIR)/, $(dir)))
TARGETDIRS = $(foreach dir, $(DIRS), $(addprefix $(BUILDDIR)/, $(dir)))

# Generate the GCC includes parameters by adding -I before each source folder
INCLUDES = $(foreach dir, $(SOURCEDIRS), $(addprefix -I, $(dir))) -I /usr/include/eigen3/Eigen

# Add this list to VPATH, the place make will look for the source files
VPATH = $(SOURCEDIRS)

# Create a list of *.c sources in DIRS
SOURCES = $(foreach dir,$(SOURCEDIRS),$(wildcard $(dir)/*.cpp))

# Define objects for all sources
OBJS := $(subst $(SOURCEDIR),$(BUILDDIR),$(SOURCES:.cpp=.o))

# Define dependencies files for all objects
DEPS = $(OBJS:.o=.d)

# Name the compiler

CC = g++ 
CFLAGS := -DMIPCH_IGNORE_CXX_SEEK -O3 -msse2 -DNDEBUG -std=c++14 -fopenmp
LDFLAGS = -lcrypto -lssl -lpthread
MAKEFLAGS += -J8

# OS specific part
RM = rm -rf 
RMDIR = rm -rf 
MKDIR = mkdir -p
ERRIGNORE = 2>/dev/null
SEP=/

# Remove space after separator
PSEP = $(strip $(SEP))

# Hide or not the calls depending of VERBOSE
ifeq ($(VERBOSE),TRUE)
    HIDE =  
else
    HIDE = @
endif

# Define the function that will generate each rule
define generateRules
$(1)/%.o: %.cpp
	@echo Building $$@
	$(HIDE)$(CC) $$(INCLUDES) -c $(CFLAGS) -o $$(subst /,$$(PSEP),$$@) $$(subst /,$$(PSEP),$$<) 
endef

.PHONY: all clean directories 

all: directories $(TARGET)

$(TARGET): $(OBJS)
	$(HIDE)echo Linking $@
	$(HIDE)$(CC) $(CFLAGS) $(OBJS) $(MAKEFLAGS) -o $(TARGET) $(LDFLAGS) 
# Include dependencies
-include $(DEPS)

# Generate rules
$(foreach targetdir, $(TARGETDIRS), $(eval $(call generateRules, $(targetdir))))

directories: 
	$(HIDE)$(MKDIR) $(subst /,$(PSEP),$(TARGETDIRS)) $(ERRIGNORE)

# Remove all objects, dependencies and executable files generated during the build
clean:
	$(HIDE)$(RMDIR) $(subst /,$(PSEP),$(TARGETDIRS)) $(ERRIGNORE)
	$(HIDE)$(RM) $(TARGET) $(ERRIGNORE)
	@echo Cleaning done ! 
