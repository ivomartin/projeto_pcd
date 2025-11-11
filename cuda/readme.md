# Configuração CUDA com Visual Studio 2022 (Versões Recentes)

## Problema

Ao compilar projetos CUDA 12.0 com versões recentes do Visual Studio 2022 (17.12 ou superior com MSVC 14.44+), você pode encontrar os seguintes erros:

1. **Erro de versão do compilador não suportada:**
   ```
   error: unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported!
   ```

2. **Erro STL1002:**
   ```
   static assertion failed with "error STL1002: Unexpected compiler version, expected CUDA 12.4 or newer."
   ```

## Causa

- **CUDA 12.0** foi lançado antes das versões mais recentes do Visual Studio 2022
- O Visual Studio 2022 17.12+ (MSVC 14.44) inclui uma STL que verifica a compatibilidade de versão
- CUDA 12.0 oficialmente suporta apenas até versões específicas do Visual Studio 2022

## Solução

Para fazer CUDA 12.0 funcionar com Visual Studio 2022 moderno, são necessárias **duas configurações** no arquivo `.vcxproj`:

### 1. Adicionar flag `-allow-unsupported-compiler`

Esta flag instrui o NVCC a prosseguir mesmo com uma versão do compilador não oficialmente suportada.

### 2. Adicionar macro `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`

Esta macro desabilita a verificação de versão da STL do Visual Studio.

## Configuração Detalhada

### Editar arquivo `.vcxproj`

Localize as seções `<ItemDefinitionGroup>` para as configurações **Debug** e **Release** e faça as seguintes modificações:

#### Para configuração Debug:

```xml
<ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
  <ClCompile>
    <WarningLevel>Level3</WarningLevel>
    <Optimization>Disabled</Optimization>
    <!-- ADICIONAR _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH aqui -->
    <PreprocessorDefinitions>WIN32;WIN64;_DEBUG;_CONSOLE;_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH;%(PreprocessorDefinitions)</PreprocessorDefinitions>
  </ClCompile>
<Link>
    <GenerateDebugInformation>true</GenerateDebugInformation>
    <SubSystem>Console</SubSystem>
    <AdditionalDependencies>cudart_static.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>
  </Link>
  <CudaCompile>
    <TargetMachinePlatform>64</TargetMachinePlatform>
    <!-- ADICIONAR estas duas linhas -->
    <AdditionalOptions>-allow-unsupported-compiler %(AdditionalOptions)</AdditionalOptions>
    <Defines>_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH;%(Defines)</Defines>
  </CudaCompile>
</ItemDefinitionGroup>
```

#### Para configuração Release:

```xml
<ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
  <ClCompile>
    <WarningLevel>Level3</WarningLevel>
    <Optimization>MaxSpeed</Optimization>
    <FunctionLevelLinking>true</FunctionLevelLinking>
 <IntrinsicFunctions>true</IntrinsicFunctions>
    <!-- ADICIONAR _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH aqui -->
    <PreprocessorDefinitions>WIN32;WIN64;NDEBUG;_CONSOLE;_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH;%(PreprocessorDefinitions)</PreprocessorDefinitions>
  </ClCompile>
  <Link>
    <GenerateDebugInformation>true</GenerateDebugInformation>
    <EnableCOMDATFolding>true</EnableCOMDATFolding>
    <OptimizeReferences>true</OptimizeReferences>
    <SubSystem>Console</SubSystem>
    <AdditionalDependencies>cudart_static.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>
  </Link>
  <CudaCompile>
  <TargetMachinePlatform>64</TargetMachinePlatform>
    <!-- ADICIONAR estas duas linhas -->
 <AdditionalOptions>-allow-unsupported-compiler %(AdditionalOptions)</AdditionalOptions>
    <Defines>_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH;%(Defines)</Defines>
  </CudaCompile>
</ItemDefinitionGroup>
```

## Passo a Passo

1. **Fechar o Visual Studio** ou **Descarregar o projeto** (clique com botão direito no projeto ? Descarregar Projeto)

2. **Editar o arquivo `.vcxproj`** com um editor de texto (Notepad++, VS Code, etc.)

3. **Localizar** as seções `<ItemDefinitionGroup>` para Debug e Release

4. **Adicionar** nas seções `<ClCompile><PreprocessorDefinitions>`:
   ```
   _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH;
   ```
   (adicione no início ou no final, antes de `%(PreprocessorDefinitions)`)

5. **Adicionar** nas seções `<CudaCompile>` as seguintes linhas:
   ```xml
   <AdditionalOptions>-allow-unsupported-compiler %(AdditionalOptions)</AdditionalOptions>
   <Defines>_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH;%(Defines)</Defines>
   ```

6. **Salvar** o arquivo

7. **Reabrir o Visual Studio** ou **Recarregar o projeto** (clique com botão direito ? Recarregar Projeto)

8. **Compilar** o projeto (Ctrl+Shift+B)

## Verificação

Se tudo estiver configurado corretamente, o projeto deve compilar sem erros e executar normalmente. 

Para o exemplo de adição de vetores, você deve ver a saída:
```
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55}
```

## Alternativas

### Opção 1: Atualizar CUDA (Recomendado para produção)

Se você está iniciando um novo projeto ou pode atualizar:
- **CUDA 12.4 ou superior** suporta oficialmente Visual Studio 2022 17.12+
- Baixe em: https://developer.nvidia.com/cuda-downloads
- Esta é a solução mais segura para ambientes de produção

### Opção 2: Usar Visual Studio 2019 ou VS 2022 mais antigo

- Visual Studio 2019 é totalmente compatível com CUDA 12.0
- Versões mais antigas do VS 2022 (antes da 17.12) também funcionam sem as flags adicionais

## Avisos Importantes

?? **Atenção:**
- Usar `-allow-unsupported-compiler` pode resultar em comportamentos inesperados em casos avançados
- Para projetos críticos ou de produção, considere usar versões oficialmente compatíveis
- Teste bem seu código, especialmente recursos avançados de CUDA
- Esta configuração funciona para a maioria dos casos de uso comum

? **Quando usar esta solução:**
- Projetos de aprendizado e experimentação
- Protótipos e desenvolvimento inicial
- Quando não é possível atualizar CUDA ou downgrade do Visual Studio
- Para código CUDA relativamente simples (como o exemplo de adição de vetores)

## Informações de Versão

- **CUDA:** 12.0
- **Visual Studio:** 2022 (17.12+, MSVC 14.44)
- **Platform Toolset:** v143
- **C++ Standard:** C++14

## Recursos Adicionais

- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Compatibility Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [Visual Studio Platform Toolset](https://learn.microsoft.com/en-us/cpp/build/how-to-modify-the-target-framework-and-platform-toolset)

---

**Última atualização:** 2024  
**Status:** ? Testado e funcionando
