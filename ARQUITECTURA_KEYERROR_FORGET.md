# Análisis Arquitectónico: KeyError 'forget' en WGA.compute_loss()

## El Problema Original

**Error:** `KeyError: 'forget'` en `src/trainer/unlearn/wga.py` línea donde accede `inputs["forget"]`

**Contexto:** El usuario tenía `remove_unused_columns=False` configurado, pero seguía perdiendo la clave "forget" antes de que llegara a `compute_loss`.

---

## Flujo de Datos (Correcto)

### 1. Dataset Output ✅
```python
ForgetRetainDataset.__getitem__(idx)  
→ Returns: {"forget": {...}, "retain": {...}}
```

### 2. Collator Processing ✅
```python
DataCollatorForSupervisedDataset.__call__(batch)

# Si recibe:
# [{"forget": {...}, "retain": {...}}, 
#  {"forget": {...}, "retain": {...}}]

# Usa recursión inteligente:
if "input_ids" not in instances[0]:  # True, porque instances[0] = {"forget": ..., "retain": ...}
    for key in instances[0].keys():  # key = "forget", "retain"
        key_instances = self.get_instances_from_key(instances, key)
        # key_instances = [instances[0]["forget"], instances[1]["forget"], ...] 
        return_dct[key] = self(key_instances)  # Llamada recursiva
        
# Segunda llamada recursiva:
# instances[0] = {"input_ids": [...], "attention_mask": [...], "labels": [...]}
# Ahora SÍ tiene "input_ids" → va a else branch
# Retorna: {"input_ids": tensor(...), "attention_mask": tensor(...), "labels": tensor(...)}

# Resultado final:
# {"forget": {"input_ids": ..., "attention_mask": ..., "labels": ...},
#  "retain": {"input_ids": ..., "attention_mask": ..., "labels": ...}}
```

### 3. El Problema: HuggingFace Trainer

Aquí es donde falla la arquitectura:

```
HF Trainer.train()
  → training_step(model, batch)
    → self._prepare_inputs(batch)  # Aquí se pierde la estructura
    → self.compute_loss(model, inputs)
```

**PROBLEMA 1: `label_names` implícito**

HuggingFace Trainer tiene una propiedad `label_names` que por defecto es:
```python
label_names = ["labels"]
```

Trainer usa esto para determinar qué es "label" vs "features". Cuando NO reconoce la estructura, intenta aplicar su lógica de extracción de labels en el nivel raíz del diccionario, esperando encontrar claves como `"input_ids"`, `"attention_mask"`, `"labels"` en el top-level.

**PROBLEMA 2: Firma implícita del modelo**

Aunque `remove_unused_columns=False`, existen OTROS mecanismos en Trainer que inspecciona la firma de la función `compute_loss` y puede estar filtrando inputs basándose en eso.

**PROBLEMA 3: `_prepare_inputs` NO preserva estructura anidada**

El método por defecto en Trainer es:
```python
def _prepare_inputs(self, inputs):
    if isinstance(inputs, (tuple, list)):
        return tuple(self._prepare_inputs(e) for e in inputs)
    elif isinstance(inputs, dict):
        return {k: self._prepare_inputs(v) for k, v in inputs.items()}  # RECURSIVO
    else:
        return inputs.to(self.args.device)
```

**Teóricamente esto debería preservar la estructura**, pero hay casos edge donde la recursión se vuelve problemática si el Trainer hace presunciones sobre qué contiene cada nivel.

---

## La Solución Implementada

### 1. Override de `label_names` 

```python
@property
def label_names(self):
    return []
```

**Por qué funciona:** Al retornar lista vacía, le decimos a Trainer: "No intentes extraer labels del diccionario raíz. Los labels están dentro de estructuras anidadas manejadas por nuestro compute_loss personalizado".

### 2. Override de `_prepare_inputs`

```python
def _prepare_inputs(self, inputs: dict):
    if isinstance(inputs, dict):
        return {k: self._prepare_inputs(v) for k, v in inputs.items()}  # Recursivo explícito
    elif isinstance(inputs, (tuple, list)):
        return tuple(self._prepare_inputs(item) for item in inputs)
    else:
        return inputs.to(self.args.device) if hasattr(inputs, 'to') else inputs
```

**Por qué funciona:** Garantiza que:
- Dicts anidados se preservan exactamente
- Solo los tensores se mueven a GPU
- NO hay aplastamiento de estructura

### 3. Debug en `compute_loss`

Agregué verificación explícita para diagnosticar si la estructura llega correctamente:

```python
if not isinstance(inputs, dict) or "forget" not in inputs:
    logger.error(f"Structure corrupted! Received: {inputs.keys()}")
    raise KeyError(...)
```

---

## Resumen: El Fallo Arquitectónico

| Fase | Status | Detalle |
|------|--------|---------|
| **Dataset Output** | ✅ OK | `{"forget": {...}, "retain": {...}}` |
| **Collator Output** | ✅ OK | Estructura anidada correcta |
| **Trainer._prepare_inputs** | ❌ FALLO | Potencialmente aplasta o presume estructura plana |
| **label_names detection** | ❌ FALLO | Trainer busca labels en top-level, no en anidado |
| **compute_loss input** | ❌ FALLO | Recibe estructura plana sin "forget"/"retain" |

**Raíz del problema:** Trainer de HF asume que las estructuras de batch son siempre planas (`{key: tensor}`), no anidadas.

**La solución:** Sobreescribir los métodos base para informarle a Trainer explícitamente que preserve la estructura anidada.

---

## Configuración Verificada

En `configs/experiment/unlearn/sequential.yaml` debe existir:

```yaml
trainer:
  args:
    remove_unused_columns: False  # ✅ Necesario (aunque no es suficiente)
    # ... otros args
```

Pero esto SOLO es el punto de partida. Los overrides en `UnlearnTrainer` son lo que realmente soluciona el problema.

---

## Testing

Después de estos cambios, el collator debería retornar:

```python
{
    "forget": {
        "input_ids": tensor([[...],...]),
        "attention_mask": tensor([[...],...]),
        "labels": tensor([[...],...])
    },
    "retain": {
        "input_ids": tensor([[...],...]),
        "attention_mask": tensor([[...],...]),
        "labels": tensor([[...],...])
    }
}
```

Y cuando llegue a `GradDiff.compute_loss()`, la verificación de debug debería pasar sin errores.
